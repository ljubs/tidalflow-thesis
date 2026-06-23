#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include "graph.h"

static constexpr int DEFAULT_BLOCK_SIZE = 256;

extern "C" int runTidalFlowCUDAPrefixInstanceWithBlockSize(const char *graphPath,
                                                     int source,
                                                     int sink,
                                                     int mode,
                                                     int blockSize,
                                                     double *totalTimeOut,
                                                     long long *maxFlowOut);

static __device__ __forceinline__ unsigned long long minULLDevice(unsigned long long a, unsigned long long b) { return (a < b) ? a : b; }

static __global__ void highTide(const int *src, const int *dst, const int *capacity,
                        const int *flow, const int *levelEdges, int levelEdgeCount,
                        unsigned long long *h, int *p) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= levelEdgeCount) return;

    int e = levelEdges[threadId];
    int u = src[e];
    int v = dst[e];
    int residual = capacity[e] - flow[e];
    if (residual <= 0) {
        p[e] = 0;
        return;
    }

    unsigned long long available = h[u];
    if (available <= 0) {
        p[e] = 0;
        return;
    }

    int promised = (int)minULLDevice((unsigned long long)residual, available);
    p[e] = promised;
    atomicAdd(&h[v], (unsigned long long)promised);
}

static __global__ void lowTide(const int *src, const int *dst, const int *levelEdges,
                        int levelStart, int levelCount, const unsigned long long *h,
                        unsigned long long *l, int *p) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= levelCount) return;

    int e = levelEdges[levelStart + (levelCount - 1 - threadId)];
    int u = src[e];
    int v = dst[e];

    int remaining = p[e];
    if (remaining <= 0) return;

    int sentTotal = 0;
    while (remaining > 0) {
        unsigned long long lV = atomicAdd(&l[v], 0ULL);
        if (lV <= 0) break;
        unsigned long long lU = atomicAdd(&l[u], 0ULL);
        unsigned long long availableU = (h[u] > lU) ? (h[u] - lU) : 0ULL;
        if (availableU <= 0) break;

        int sendBack = (int)minULLDevice((unsigned long long)remaining, minULLDevice(lV, availableU));
        if (sendBack <= 0) break;

        if (atomicCAS(&l[v], lV, lV - (unsigned long long)sendBack) == lV) {
            if (atomicCAS(&l[u], lU, lU + (unsigned long long)sendBack) == lU) {
                sentTotal += sendBack;
                remaining -= sendBack;
            } else {
                atomicAdd(&l[v], (unsigned long long)sendBack);
            }
        }
    }

    p[e] = sentTotal;
}

static __global__ void erosion(const int *src, const int *dst, const int *rev, const int *levelEdges,
                        int levelStart, int levelCount, unsigned long long *h, int *p, int *flow) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= levelCount) return;

    int e = levelEdges[levelStart + threadId];
    int u = src[e];
    int v = dst[e];
    int reverseEdge = rev[e];

    int remaining = p[e];
    if (remaining <= 0) return;

    int commitTotal = 0;
    while (remaining > 0) {
        unsigned long long hU = atomicAdd(&h[u], 0ULL);
        if (hU <= 0) break;

        int commitFlow = (int)minULLDevice((unsigned long long)remaining, hU);
        if (commitFlow <= 0) break;

        if (atomicCAS(&h[u], hU, hU - (unsigned long long)commitFlow) == hU) {
            atomicAdd(&h[v], (unsigned long long)commitFlow);
            flow[e] += commitFlow;
            flow[reverseEdge] -= commitFlow;

            commitTotal += commitFlow;
            remaining -= commitFlow;
        }
    }

    p[e] = commitTotal;
}

template <int BLOCK_SIZE>
static long long tideCycle(Graph *g, int source, int sink, const int *levelVer, int numLevels,
                     int *d_src, int *d_dst, int *d_rev, int *d_capacity, int *d_flow,
                     int *d_levelEdges, unsigned long long *d_h, int *d_p, unsigned long long *d_l) {
    const size_t nodeFlowBytes = (size_t)g->nodeCount * sizeof(unsigned long long);

    // high tide
    // high tide overwrites active p entries, the stale ones remain but this doesn't matter
    // cudaMemset(d_p, 0, g->edgeCount * sizeof(int));
    cudaMemset(d_h, 0, nodeFlowBytes);
    unsigned long long inf = ULLONG_MAX;
    cudaMemcpy(d_h + source, &inf, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    for (int level = 0; level < numLevels; level++) {
        int levelStart = levelVer[level];
        int levelEnd = levelVer[level + 1];
        int levelCount = levelEnd - levelStart;
        if (levelCount <= 0) continue;
        int numBlocks = (levelCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        highTide<<<numBlocks, BLOCK_SIZE>>>(
            d_src, d_dst, d_capacity, d_flow, d_levelEdges + levelStart, levelCount, d_h, d_p);
    }

    unsigned long long sinkHighTide = 0;
    cudaMemcpy(&sinkHighTide, d_h + sink, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (sinkHighTide == 0) return 0;

    // low tide
    cudaMemset(d_l, 0, nodeFlowBytes);
    cudaMemcpy(d_l + sink, &sinkHighTide, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    for (int level = numLevels - 1; level >= 0; level--) {
        int levelStart = levelVer[level];
        int levelEnd = levelVer[level + 1];
        int levelCount = levelEnd - levelStart;
        if (levelCount <= 0) continue;
        int numBlocks = (levelCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        lowTide<<<numBlocks, BLOCK_SIZE>>>(
            d_src, d_dst, d_levelEdges, levelStart, levelCount, d_h, d_l, d_p);
    }

    unsigned long long sourceLowTide = 0;
    cudaMemcpy(&sourceLowTide, d_l + source, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (sourceLowTide <= 0) return 0;

    // erosion
    cudaMemset(d_h, 0, nodeFlowBytes);
    cudaMemcpy(d_h + source, &sourceLowTide, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    for (int level = 0; level < numLevels; level++) {
        int levelStart = levelVer[level];
        int levelEnd = levelVer[level + 1];
        int levelCount = levelEnd - levelStart;
        if (levelCount <= 0) continue;
        int numBlocks = (levelCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        erosion<<<numBlocks, BLOCK_SIZE>>>(
            d_src, d_dst, d_rev, d_levelEdges, levelStart, levelCount, d_h, d_p, d_flow);
    }

    unsigned long long flowSent = 0;
    cudaMemcpy(&flowSent, d_h + sink, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    return (long long)flowSent;
}

template <int BLOCK_SIZE>
static __global__ void buildLevelGraphKernel(const int *ver, const int *dst, const int *capacity,
                                      const int *flow, const int *frontier, int frontierSize,
                                      int nextLevel, int *dist, int *nextFrontier,
                                      int *nextFrontierSize, int *levelEdges, int *levelEdgeCount,
                                      int *edgeThatDiscoveredVertex) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // local to each thread
    int localFrontierCount = 0;
    int localLevelEdgeCount = 0;

    typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan; // slags data struktur som lar oss gjøre prefix sum over en verdi per tråd i en blokk
    __shared__ typename BlockScan::TempStorage frontierCountScanStorage; // midlertidig shared memory som BlockScan trenger for å jobbe
    __shared__ typename BlockScan::TempStorage levelEdgeCountScanStorage;
    __shared__ int blockFrontierBase;
    __shared__ int blockLevelEdgeBase;

    // finner noder og nivåkanter
    if (threadId < frontierSize) { // threads outside of this range just wait
        int u = frontier[threadId];
        for (int e = ver[u]; e < ver[u + 1]; e++) {
            int v = dst[e];
            int residual = capacity[e] - flow[e];
            if (residual <= 0) continue;

            int oldDist = atomicCAS(&dist[v], -1, nextLevel);
            if (oldDist == -1) {
                edgeThatDiscoveredVertex[v] = e;
                localFrontierCount++;
                localLevelEdgeCount++;
            } else if (oldDist == nextLevel)
                localLevelEdgeCount++;
        }
    }

    int frontierOffset = 0; // where this thread starts writing inside the block
    int levelEdgeOffset = 0;
    int blockFrontierCount = 0; // hvor mange hele blokken fant
    int blockLevelEdgeCount = 0;

    // leser localFrontierCount fra hver tråd, skriver frontierOffset for hver tråd,
    // og blockFrontierCount blir totalen for hele blokken
    // så basically scanner gjennom hvor mange noder/kanter hver tråd fant og plusser forrige prefix sum -> denne noden begynner å skrive fra denne offset'en
    BlockScan(frontierCountScanStorage).ExclusiveSum(localFrontierCount, frontierOffset, blockFrontierCount);
    BlockScan(levelEdgeCountScanStorage).ExclusiveSum(localLevelEdgeCount, levelEdgeOffset, blockLevelEdgeCount);


    // her lagrer vi hvor vi skal skrive i next frontier == blockFrontierBase, også øker vi dem med blockFrontierCount
    if (threadIdx.x == 0) {
        // remember, atomicAdd returns the old val, so...
        // ... blockFrontierBase = nextFrontierSize, then does nextFrontierSize + blockFrontierCount
        blockFrontierBase = atomicAdd(nextFrontierSize, blockFrontierCount);
        blockLevelEdgeBase = atomicAdd(levelEdgeCount, blockLevelEdgeCount);
    }

    __syncthreads();

    if (threadId < frontierSize) {
        int u = frontier[threadId];
        // vi har en base for hele blocken... også har hver tråd sin egen offset, legg det sammen og der skal de skrive i globale arrayen
        int frontierWrite = blockFrontierBase + frontierOffset;
        int levelEdgeWrite = blockLevelEdgeBase + levelEdgeOffset;

        for (int e = ver[u]; e < ver[u + 1]; e++) {
            int v = dst[e];
            int residual = capacity[e] - flow[e];
            if (residual <= 0 || dist[v] != nextLevel) continue;

            // kun "kanten" som oppdaget noden får skrive den inn, hvis ikke kunne alle kanter
            // som går inn i noden legge den inn i globale arrayen
            if (edgeThatDiscoveredVertex[v] == e)
                nextFrontier[frontierWrite++] = v;

            levelEdges[levelEdgeWrite++] = e;
        }
    }
}

static __global__ void backwardReachKernel(const int *src, const int *dst, const int *levelEdges,
                                    int levelStart, int levelCount, int *reach) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= levelCount) return;

    int e = levelEdges[levelStart + threadId];
    int v = dst[e];
    if (reach[v]) {
        int u = src[e]; // lazy load u
        reach[u] = 1;
    }
}

static __global__ void compactLevelEdgesKernel(const int *src, const int *dst, const int *inLevelEdges,
                                        int levelStart, int levelCount, const int *reach,
                                        int *outLevelEdges, int *writeCount) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= levelCount) return;

    int e = inLevelEdges[levelStart + threadId];
    int u = src[e];
    int v = dst[e];
    if (reach[u] && reach[v]) {
        int pos = atomicAdd(writeCount, 1);
        outLevelEdges[pos] = e;
    }
}

template <int BLOCK_SIZE>
static int buildLevelGraph(Graph *g, int source, int sink, int *levelVer,
                           int *numLevels, const int *d_ver, const int *d_src,
                           const int *d_dst, const int *d_capacity, const int *d_flow,
                           int *d_levelEdges, int *d_compactedLevelEdges, int *d_dist, int *d_frontier,
                           int *d_nextFrontier, int *d_levelEdgeCount, int *d_nextFrontierSize,
                           int *d_reach, int *d_writeCount, int *d_frontierOwnerEdge,
                           int *oldLevelVer) {
    const size_t nodeBytes = (size_t)g->nodeCount * sizeof(int);
    int level = 0;
    int frontierSize = 1;
    int sinkDist = -1;
    int sourceReach = 0;
    int sourceDist = 0;
    int one = 1;
    int edgeCount = 0;
    int hasLevel = 0;

    cudaMemset(d_dist, 0xFF, nodeBytes);

    cudaMemcpy(d_dist + source, &sourceDist, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_levelEdgeCount, 0, sizeof(int));

    levelVer[0] = 0;

    int *currFrontier = d_frontier;
    int *nextFrontier = d_nextFrontier;

    while (frontierSize > 0) {
        int nextLevel = level + 1;
        int numBlocks = (frontierSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMemset(d_nextFrontierSize, 0, sizeof(int));
        cudaMemset(d_frontierOwnerEdge, 0xFF, nodeBytes);

        buildLevelGraphKernel<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE>>>(
            d_ver, d_dst, d_capacity, d_flow, currFrontier, frontierSize, nextLevel,
            d_dist, nextFrontier, d_nextFrontierSize, d_levelEdges, d_levelEdgeCount,
            d_frontierOwnerEdge);

        cudaMemcpy(&frontierSize, d_nextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost);

        cudaMemcpy(&edgeCount, d_levelEdgeCount, sizeof(int), cudaMemcpyDeviceToHost);
        level++;
        levelVer[level] = edgeCount;

        cudaMemcpy(&sinkDist, d_dist + sink, sizeof(int), cudaMemcpyDeviceToHost);
        if (sinkDist != -1) break;

        int *tmp = currFrontier;
        currFrontier = nextFrontier;
        nextFrontier = tmp;
    }

    if (sinkDist == -1) {
        *numLevels = level;
        return 0;
    }

    memcpy(oldLevelVer, levelVer, (level + 1) * sizeof(int));

    cudaMemset(d_reach, 0, nodeBytes);

    cudaMemcpy(d_reach + sink, &one, sizeof(int), cudaMemcpyHostToDevice);

    for (int lvl = level - 1; lvl >= 0; lvl--) {
        int start = oldLevelVer[lvl];
        int end = oldLevelVer[lvl + 1];
        int count = end - start;
        if (count == 0) continue;

        int numBlocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        backwardReachKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_src, d_dst, d_levelEdges, start, count, d_reach);
    }

    cudaMemset(d_writeCount, 0, sizeof(int));
    levelVer[0] = 0;
    for (int lvl = 0; lvl < level; lvl++) {
        int start = oldLevelVer[lvl];
        int end = oldLevelVer[lvl + 1];
        int count = end - start;
        if (count > 0) {
            int numBlocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            compactLevelEdgesKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_src, d_dst, d_levelEdges, start, count, d_reach, d_compactedLevelEdges, d_writeCount);
        }
        cudaMemcpy(&edgeCount, d_writeCount, sizeof(int), cudaMemcpyDeviceToHost);
        levelVer[lvl + 1] = edgeCount;
    }

    cudaMemcpy(&sourceReach, d_reach + source, sizeof(int), cudaMemcpyDeviceToHost);

    *numLevels = level;

    hasLevel = (edgeCount > 0 && sourceReach != 0);

    return hasLevel;
}

template <int BLOCK_SIZE>
static long long tidalFlowImpl(Graph *g, int source, int sink) {
    long long maxFlow = 0;
    int numLevels = 0;
    const size_t nodeBytes = (size_t)g->nodeCount * sizeof(int);
    const size_t nodeFlowBytes = (size_t)g->nodeCount * sizeof(unsigned long long);
    const size_t nodeVerBytes = (size_t)(g->nodeCount + 1) * sizeof(int);
    const size_t edgeBytes = (size_t)g->edgeCount * sizeof(int);
    int *levelVer = (int *)malloc(nodeVerBytes);
    int *oldLevelVer = (int *)malloc(nodeVerBytes);

    int *d_ver = NULL, *d_src = NULL, *d_dst = NULL, *d_rev = NULL, *d_capacity = NULL, *d_flow = NULL;
    int *d_levelEdges = NULL, *d_compactedLevelEdges = NULL, *d_p = NULL;
    unsigned long long *d_h = NULL, *d_l = NULL;
    int *d_dist = NULL, *d_frontier = NULL, *d_nextFrontier = NULL;
    int *d_levelEdgeCount = NULL, *d_nextFrontierSize = NULL, *d_frontierOwnerEdge = NULL;
    int *d_writeCount = NULL;
    cudaMalloc(&d_ver, nodeVerBytes);
    cudaMalloc(&d_src, edgeBytes);
    cudaMalloc(&d_dst, edgeBytes);
    cudaMalloc(&d_rev, edgeBytes);
    cudaMalloc(&d_capacity, edgeBytes);
    cudaMalloc(&d_flow, edgeBytes);
    cudaMalloc(&d_levelEdges, edgeBytes);
    cudaMalloc(&d_compactedLevelEdges, edgeBytes);
    cudaMalloc(&d_dist, nodeBytes);
    cudaMalloc(&d_frontier, nodeBytes);
    cudaMalloc(&d_nextFrontier, nodeBytes);
    cudaMalloc(&d_frontierOwnerEdge, nodeBytes);
    cudaMalloc(&d_h, nodeFlowBytes);
    cudaMalloc(&d_l, nodeFlowBytes);
    cudaMalloc(&d_levelEdgeCount, sizeof(int));
    cudaMalloc(&d_nextFrontierSize, sizeof(int));
    cudaMalloc(&d_writeCount, sizeof(int));

    d_p = d_levelEdges; // reuse after compaction

    cudaMemcpy(d_ver, g->ver, nodeVerBytes, cudaMemcpyHostToDevice);

    cudaMemcpy(d_src, g->src, edgeBytes, cudaMemcpyHostToDevice);

    cudaMemcpy(d_dst, g->dst, edgeBytes, cudaMemcpyHostToDevice);

    cudaMemcpy(d_rev, g->rev, edgeBytes, cudaMemcpyHostToDevice);

    cudaMemcpy(d_capacity, g->capacity, edgeBytes, cudaMemcpyHostToDevice);

    cudaMemset(d_flow, 0, edgeBytes);

    while (1) {
        int hasLevel = buildLevelGraph<BLOCK_SIZE>(g, source, sink, levelVer,
                                                   &numLevels, d_ver, d_src,
                                                   d_dst, d_capacity, d_flow, d_levelEdges,
                                                   d_compactedLevelEdges, d_dist, d_frontier, d_nextFrontier,
                                                   d_levelEdgeCount, d_nextFrontierSize, d_dist, d_writeCount,
                                                   d_frontierOwnerEdge, oldLevelVer);
        if (!hasLevel) break;
        while (1) {
            long long flow = tideCycle<BLOCK_SIZE>(g, source, sink, levelVer, numLevels,
                                                   d_src, d_dst, d_rev, d_capacity, d_flow,
                                                   d_compactedLevelEdges, d_h, d_p, d_l);
            if (flow == 0) break;
            maxFlow += flow;
        }
    }
    cudaFree(d_ver);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_rev);
    cudaFree(d_capacity);
    cudaFree(d_flow);
    cudaFree(d_levelEdges);
    cudaFree(d_compactedLevelEdges);
    cudaFree(d_dist);
    cudaFree(d_frontier);
    cudaFree(d_nextFrontier);
    cudaFree(d_frontierOwnerEdge);
    cudaFree(d_h);
    cudaFree(d_l);
    cudaFree(d_levelEdgeCount);
    cudaFree(d_nextFrontierSize);
    cudaFree(d_writeCount);

    free(oldLevelVer);
    free(levelVer);

    return maxFlow;
}

static long long tidalFlowWithBlockSize(Graph *g, int source, int sink, int blockSize) {
    switch (blockSize) {
        case 128:
            return tidalFlowImpl<128>(g, source, sink);
        case 256:
            return tidalFlowImpl<256>(g, source, sink);
        case 512:
            return tidalFlowImpl<512>(g, source, sink);
        case 1024:
            return tidalFlowImpl<1024>(g, source, sink);
        default:
            return -1;
    }
}

static int isSupportedBlockSize(int blockSize) {
    return blockSize == 128 || blockSize == 256 || blockSize == 512 || blockSize == 1024;
}

long long tidalFlowCUDAPrefix(Graph *g, int source, int sink) {
    return tidalFlowWithBlockSize(g, source, sink, DEFAULT_BLOCK_SIZE);
}

extern "C" int runTidalFlowCUDAPrefixInstance(const char *graphPath,
                                        int source,
                                        int sink,
                                        int mode,
                                        double *totalTimeOut,
                                        long long *maxFlowOut) {
    return runTidalFlowCUDAPrefixInstanceWithBlockSize(
        graphPath, source, sink, mode, DEFAULT_BLOCK_SIZE, totalTimeOut, maxFlowOut);
}

extern "C" int runTidalFlowCUDAPrefixInstanceWithBlockSize(const char *graphPath,
                                                     int source,
                                                     int sink,
                                                     int mode,
                                                     int blockSize,
                                                     double *totalTimeOut,
                                                     long long *maxFlowOut) {
    if (!isSupportedBlockSize(blockSize)) return -1;

    Graph g;
    if (buildGraphFromECLFileWithMode(&g, graphPath, source, mode) != 0) {
        return -1;
    }

    if (source < 0 || source >= g.nodeCount || sink < 0 || sink >= g.nodeCount) {
        freeGraph(&g);
        return -1;
    }

    cudaFree(0);

    double start = omp_get_wtime();
    long long maxFlow = tidalFlowWithBlockSize(&g, source, sink, blockSize);
    double end = omp_get_wtime();

    if (totalTimeOut != NULL) *totalTimeOut = end - start;
    if (maxFlowOut != NULL) *maxFlowOut = maxFlow;

    freeGraph(&g);
    return 0;
}
