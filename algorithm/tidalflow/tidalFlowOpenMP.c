#include <limits.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "graph.h"

static inline long long minLongLong(long long a, long long b) { return (a < b) ? a : b; }

static inline int atomicCASInt(int *ptr, int expected, int desired) {
    int old = expected;
    __atomic_compare_exchange_n(ptr, &old, desired, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED); // GCC/Clang sin innebygde CAS operasjon
    return old;
}

// etterhvert så bruker vi jo cas på h[u] som kan bli veldig stor, så trenger egen cas
static inline long long atomicCASLongLong(long long *ptr, long long expected, long long desired) {
    long long old = expected;
    __atomic_compare_exchange_n(ptr, &old, desired, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
    return old;
}

static void markPrunedReach(Graph *g, const int *levelEdges, const int *levelVer,
                            int level, int sink, int *reach) {
    memset(reach, 0, (size_t)g->nodeCount * sizeof(int));
    reach[sink] = 1;

    for (int lvl = level - 1; lvl >= 0; lvl--) {
        int start = levelVer[lvl];
        int end = levelVer[lvl + 1];

        #pragma omp parallel for schedule(static)
        for (int i = end - 1; i >= start; i--) {
            int e = levelEdges[i];
            int u = g->src[e];
            int v = g->dst[e];
            if (reach[v]) {
                #pragma omp atomic write // strengt tatt ikke nødvendig... går fint om flere skriver 1 her
                reach[u] = 1;
            }
        }
    }
}

static int compactLevelEdgesParallel(Graph *g, const int *levelEdges, int *compactLevelEdges,
                                     int *oldLevelVer, int *levelVer, int level, const int *reach) {
    int writeCount = 0;
    levelVer[0] = 0;

    for (int lvl = 0; lvl < level; lvl++) {
        int start = oldLevelVer[lvl];
        int end = oldLevelVer[lvl + 1];

        #pragma omp parallel for schedule(static)
        for (int i = start; i < end; i++) {
            int e = levelEdges[i];
            int u = g->src[e];
            int v = g->dst[e];

            if (reach[u] && reach[v]) {
                int pos;
                /*
                husk:
                atomic capture gjør to ting
                låser {
                    pos = writeCount; // fanger posisjonen den skal skrive de filtrerte nivåkantene på
                    writeCount++; // også øker den writeCount for neste tråd som skal skrive
                }
                */
                #pragma omp atomic capture
                pos = writeCount++;
                compactLevelEdges[pos] = e;
            }
        }

        levelVer[lvl + 1] = writeCount;
    }

    return writeCount;
}

static int buildLevelGraph(Graph *g, int source, int sink,
                           int levelEdges[], int compactLevelEdges[],
                           int *dist, int *global, int *foundNodes, int *foundLevelEdges,
                           int *localNodes, int *localLevelEdges,
                           int *levelVer, int *numLevels) {
    memset(dist, -1, (size_t)g->nodeCount * sizeof(int));
    dist[source] = 0;

    int globalSize = 0;
    global[globalSize++] = source;

    int levelEdgeCount = 0;
    int level = 0;
    levelVer[0] = 0;

    while (globalSize != 0) {
        int newSize = 0;
        int levelEdgeBase = levelEdgeCount;

        #pragma omp parallel
        {
            int threadId = omp_get_thread_num();
            int *myLocalNodes = &localNodes[(size_t)threadId * (size_t)g->nodeCount];
            int localNodesSize = 0;
            int *myLocalLevelEdges = &localLevelEdges[(size_t)threadId * (size_t)g->edgeCount];
            int localLevelEdgesSize = 0;

            #pragma omp for nowait
            for (int i = 0; i < globalSize; i++) {
                int u = global[i];
                int nextLevel = dist[u] + 1;

                for (int e = g->ver[u]; e < g->ver[u + 1]; e++) {
                    int v = g->dst[e];
                    int residual = g->capacity[e] - g->flow[e];
                    if (residual <= 0) continue;

                    int distV;
                    #pragma omp atomic read
                    distV = dist[v];
                    if (distV == -1) { // denne her trenger vi strengt tatt ikke
                        if (atomicCASInt(&dist[v], -1, nextLevel) == -1) {
                            myLocalNodes[localNodesSize++] = v;
                            distV = nextLevel;
                        } else { // i tilfelle den endret seg etter at vi sjekket at distV == -1
                            #pragma omp atomic read
                            distV = dist[v];
                        }
                    }

                    if (distV == nextLevel) {
                        myLocalLevelEdges[localLevelEdgesSize++] = e;
                    }
                }
            }

            foundNodes[threadId] = localNodesSize; // forteller hvor mange noder den fant
            foundLevelEdges[threadId] = localLevelEdgesSize; // forteller hvor mange kanter den fant
            #pragma omp barrier

            // PREFIX SUM STARTER HER
            int offsetNodes = 0;
            int offsetLevelEdges = levelEdgeBase;
            for (int t = 0; t < threadId; t++) {
                offsetNodes += foundNodes[t];
                offsetLevelEdges += foundLevelEdges[t];
            } // teller opp gjennom prefix sum

            if (threadId == omp_get_num_threads() - 1) { // siste tråden regner ut størrelsen på frontieren, fordi han regner ut totalen i prefix summen
                // MÅ VÆRE SISTE
                newSize = offsetNodes + localNodesSize;
                levelEdgeCount = offsetLevelEdges + localLevelEdgesSize;
            }

            // hver tråd tar sin memcpy nå, og kopierer over noder/kanter de fant til f.eks. global[offsetNodes] <- altså deres posisjon i global
            memcpy(&global[offsetNodes], myLocalNodes, (size_t)localNodesSize * sizeof(int));
            memcpy(&levelEdges[offsetLevelEdges], myLocalLevelEdges, (size_t)localLevelEdgesSize * sizeof(int));
        }

        globalSize = newSize;
        level++;
        levelVer[level] = levelEdgeCount; // regner ut hvor mange kanter vi fant i dette nivået
        if (dist[sink] != -1) break;
    }

    if (dist[sink] == -1) {
        *numLevels = level;
        return 0;
    }

    int *reach = localNodes;
    markPrunedReach(g, levelEdges, levelVer, level, sink, reach);

    int *oldLevelVer = global;
    memcpy(oldLevelVer, levelVer, (size_t)(level + 1) * sizeof(int));

    int write = compactLevelEdgesParallel(g, levelEdges, compactLevelEdges, oldLevelVer, levelVer, level, reach);

    *numLevels = level;
    return (write > 0 && reach[source] != 0); // sjekker om vi kan nå sluket fra kilden, hvis ikke -> return false
}

static long long tideCycle(Graph *g, int source, int sink, int *levelEdges,
                           long long *h, int *p, long long *l, int *levelVer, int numLevels) {
    memset(h, 0, (size_t)g->nodeCount * sizeof(long long));
    h[source] = LLONG_MAX;
    for (int level = 0; level < numLevels; level++) {
        int levelStart = levelVer[level];
        int levelEnd = levelVer[level + 1];

        #pragma omp parallel for schedule(static)
        for (int i = levelStart; i < levelEnd; i++) {
            int edge = levelEdges[i];
            int u = g->src[edge];
            int v = g->dst[edge];
            int residual = g->capacity[edge] - g->flow[edge];
            int promised = (int)minLongLong((long long)residual, h[u]);
            p[i] = promised;
            #pragma omp atomic update
            h[v] += promised;
        }
    }
    if (h[sink] == 0) return 0;

    memset(l, 0, (size_t)g->nodeCount * sizeof(long long));
    l[sink] = h[sink];
    for (int level = numLevels - 1; level >= 0; level--) {
        int levelStart = levelVer[level];
        int levelEnd = levelVer[level + 1];

        #pragma omp parallel for schedule(static)
        for (int i = levelEnd - 1; i >= levelStart; i--) {
            int edge = levelEdges[i];
            int u = g->src[edge];
            int v = g->dst[edge];

            int sentTotal = 0;
            int remaining = p[i];
            while (remaining > 0) {
                long long lV;
                #pragma omp atomic read
                lV = l[v];
                if (lV <= 0) break;

                long long lU;
                #pragma omp atomic read
                lU = l[u];
                long long availableU = h[u] - lU;
                if (availableU <= 0) break;

                int sendBack = (int)minLongLong((long long)remaining, minLongLong(lV, availableU));
                if (sendBack <= 0) break;

                /*
                hvorfor remaining: vi lagrer hvor mye vi kan sende gjennom den kanten p[e]
                - også er det mulig at vi sender i små biter, fordi vi kan oppdage at det er lite lV i første omgang, også senere finne ut at en tråd la tilbake litt flyt
                - og dermed så kan vi prøve igjen, men aldri mer enn remaining, eller noen av de andre flaskehalsene i min(...)
                */
                if (atomicCASLongLong(&l[v], lV, lV - sendBack) == lV) {
                    if (atomicCASLongLong(&l[u], lU, lU + sendBack) == lU) {
                        sentTotal += sendBack;
                        remaining -= sendBack;
                    } else {
                        #pragma omp atomic update
                        l[v] += sendBack;
                    }
                }
            }

            p[i] = sentTotal;
        }
    }

    memset(h, 0, (size_t)g->nodeCount * sizeof(long long));
    h[source] = l[source];
    for (int level = 0; level < numLevels; level++) {
        int levelStart = levelVer[level];
        int levelEnd = levelVer[level + 1];

        #pragma omp parallel for schedule(static)
        for (int i = levelStart; i < levelEnd; i++) {
            int edge = levelEdges[i];
            int u = g->src[edge];
            int v = g->dst[edge];
            int reverseEdge = g->rev[edge];

            // int commitTotal = 0;
            int remaining = p[i];
            while (remaining > 0) { // ikke nødvendig egentlig, bruker bare samme mønster som i low tide, men her legger vi ikke tilbake så vi sender enten alt eller prøver igjen
                long long hU;
                #pragma omp atomic read
                hU = h[u];
                if (hU <= 0) break;

                int commitFlow = (int)minLongLong((long long)remaining, hU);
                if (commitFlow <= 0) break;

                if (atomicCASLongLong(&h[u], hU, hU - commitFlow) == hU) {
                    #pragma omp atomic update
                    h[v] += commitFlow;
                    g->flow[reverseEdge] -= commitFlow;
                    g->flow[edge] += commitFlow;

                    // commitTotal += commitFlow;
                    remaining -= commitFlow;
                }
            }

            // p[i] = commitTotal;
        }
    }

    return h[sink];
}

long long tidalFlowOpenMP(Graph *g, int source, int sink) {
    int maxThreads = omp_get_max_threads();
    size_t nodes = (size_t)g->nodeCount;
    size_t edges = (size_t)g->edgeCount;
    size_t threadCount = (size_t)maxThreads;

    int *levelEdges = (int *)malloc(edges * sizeof(int));
    int *levelEdgesTmp = (int *)malloc(edges * sizeof(int));
    int *dist = (int *)malloc(nodes * sizeof(int));
    int *global = (int *)malloc(nodes * sizeof(int));
    int *foundNodes = (int *)malloc(threadCount * sizeof(int));
    int *foundLevelEdges = (int *)malloc(threadCount * sizeof(int));
    int *localNodes = (int *)malloc(threadCount * nodes * sizeof(int));
    int *localLevelEdges = (int *)malloc(threadCount * edges * sizeof(int));
    long long *h = (long long *)malloc(nodes * sizeof(long long));
    int *p = (int *)malloc(edges * sizeof(int));
    long long *l = (long long *)malloc(nodes * sizeof(long long));
    int *levelVer = (int *)malloc((nodes + 1) * sizeof(int));

    if (levelEdges == NULL || levelEdgesTmp == NULL || dist == NULL || global == NULL ||
        foundNodes == NULL || foundLevelEdges == NULL || localNodes == NULL ||
        localLevelEdges == NULL || h == NULL || p == NULL || l == NULL || levelVer == NULL) {
        free(levelEdges);
        free(levelEdgesTmp);
        free(dist);
        free(global);
        free(foundNodes);
        free(foundLevelEdges);
        free(localNodes);
        free(localLevelEdges);
        free(h);
        free(p);
        free(l);
        free(levelVer);
        return -1;
    }

    long long maxFlow = 0;
    int numLevels = 0;

    while (1) {
        int hasLevel = buildLevelGraph(g, source, sink,
                                       levelEdges, levelEdgesTmp,
                                       dist, global, foundNodes, foundLevelEdges,
                                       localNodes, localLevelEdges,
                                       levelVer, &numLevels);
        if (!hasLevel) break;

        while (1) {
            long long flow = tideCycle(g, source, sink, levelEdgesTmp,
                                       h, p, l, levelVer, numLevels);
            if (flow <= 0) break;
            maxFlow += flow;
        }
    }

    free(levelEdges);
    free(levelEdgesTmp);
    free(dist);
    free(global);
    free(foundNodes);
    free(foundLevelEdges);
    free(localNodes);
    free(localLevelEdges);
    free(h);
    free(p);
    free(l);
    free(levelVer);
    return maxFlow;
}
