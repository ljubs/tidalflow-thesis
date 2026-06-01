#define _POSIX_C_SOURCE 200809L

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "graph.h"
#include "tidalFlowSeqStats.h"

/*
 * this is the sequential blocking-flow variant.
 * it builds one bfs level graph, then runs tide cycles until no more flow is sent.
 */

static inline long long minLongLong(long long a, long long b) { return (a < b) ? a : b; }

static int buildLevelGraphBaseline(Graph *g, int source, int sink,
                                   int *levelEdges, int *levelEdgeCount, int *dist,
                                   int *queue) {
    memset(dist, -1, (size_t)g->nodeCount * sizeof(*dist));

    int head = 0;
    int tail = 0;

    *levelEdgeCount = 0;

    dist[source] = 0;
    queue[tail++] = source;

    while (head < tail) {
        int levelEnd = tail;

        while (head < levelEnd) {
            int u = queue[head++];
            int nextLevel = dist[u] + 1;

            for (int e = g->ver[u]; e < g->ver[u + 1]; e++) {
                int v = g->dst[e];
                int residual = g->capacity[e] - g->flow[e];

                if (residual <= 0) continue;

                if (dist[v] == -1) {
                    dist[v] = nextLevel;
                    queue[tail++] = v;
                }

                if (dist[v] == nextLevel)
                    levelEdges[(*levelEdgeCount)++] = e;
            }
        }

        if (dist[sink] != -1) break;
    }

    return dist[sink] != -1;
}

static long long tideCycleBaseline(Graph *g, int source, int sink,
                                   int *levelEdges, int levelEdgeCount, long long *h,
                                   int *p, long long *l, TidalFlowSeqStats *stats) {
    double phaseStart = tidalFlowSeqNowSeconds();
    memset(h, 0, (size_t)g->nodeCount * sizeof(long long));
    memset(l, 0, (size_t)g->nodeCount * sizeof(long long));

    h[source] = LLONG_MAX;
    for (int idx = 0; idx < levelEdgeCount; idx++) {
        int edge = levelEdges[idx];
        int u = g->src[edge];
        int v = g->dst[edge];
        int residual = g->capacity[edge] - g->flow[edge];

        p[idx] = (int)minLongLong((long long)residual, h[u]);
        h[v] += p[idx];
    }
    if (stats != NULL) stats->highTideSeconds += tidalFlowSeqNowSeconds() - phaseStart;

    if (h[sink] == 0) return 0;

    phaseStart = tidalFlowSeqNowSeconds();
    l[sink] = h[sink];
    for (int idx = levelEdgeCount - 1; idx >= 0; idx--) {
        int edge = levelEdges[idx];
        int u = g->src[edge];
        int v = g->dst[edge];
        long long available = h[u] - l[u];
        if (available < 0) {
            available = 0;
        }
        int pushed = (int)minLongLong((long long)p[idx], minLongLong(available, l[v]));

        p[idx] = pushed;
        l[v] -= pushed;
        l[u] += pushed;
    }
    if (stats != NULL) stats->lowTideSeconds += tidalFlowSeqNowSeconds() - phaseStart;

    phaseStart = tidalFlowSeqNowSeconds();
    memset(h, 0, (size_t)g->nodeCount * sizeof(long long));
    h[source] = l[source];
    for (int idx = 0; idx < levelEdgeCount; idx++) {
        int edge = levelEdges[idx];
        int u = g->src[edge];
        int v = g->dst[edge];
        int sent = (int)minLongLong((long long)p[idx], h[u]);
        int rev = g->rev[edge];

        p[idx] = sent;
        h[u] -= sent;
        h[v] += sent;
        g->flow[edge] += sent;
        g->flow[rev] -= sent;
    }
    if (stats != NULL) stats->erosionSeconds += tidalFlowSeqNowSeconds() - phaseStart;

    return h[sink];
}

long long tidalFlowSeq_BLOCKING_STATS(Graph *g, int source, int sink, TidalFlowSeqStats *stats) {
    resetTidalFlowSeqStats(stats);

    const size_t nodeBytes = (size_t)g->nodeCount * sizeof(int);
    const size_t edgeBytes = (size_t)g->edgeCount * sizeof(int);
    const size_t tideBytes = (size_t)g->nodeCount * sizeof(long long);

    int *levelEdges = (int *)malloc(edgeBytes);
    int *dist = (int *)malloc(nodeBytes);
    int *queue = (int *)malloc(nodeBytes);
    long long *h = (long long *)malloc(tideBytes);
    int *p = (int *)malloc(edgeBytes);
    long long *l = (long long *)malloc(tideBytes);

    if (levelEdges == NULL || dist == NULL || queue == NULL || h == NULL || p == NULL || l == NULL) {
        free(levelEdges);
        free(dist);
        free(queue);
        free(h);
        free(p);
        free(l);
        return -1;
    }

    long long maxFlow = 0;
    int levelEdgeCount = 0;
    double phaseStart = tidalFlowSeqNowSeconds();
    int hasLevelGraph = buildLevelGraphBaseline(g, source, sink, levelEdges, &levelEdgeCount, dist, queue);
    if (stats != NULL) {
        stats->bfsCount++;
        stats->buildLevelGraphSeconds += tidalFlowSeqNowSeconds() - phaseStart;
    }

    while (hasLevelGraph) {
        while (1) {
            phaseStart = tidalFlowSeqNowSeconds();
            long long flow = tideCycleBaseline(g, source, sink, levelEdges, levelEdgeCount, h, p, l, stats);
            if (stats != NULL) {
                stats->tideCycleCount++;
                stats->tideCycleSeconds += tidalFlowSeqNowSeconds() - phaseStart;
            }
            if (flow <= 0) break;
            maxFlow += flow;
        }
        phaseStart = tidalFlowSeqNowSeconds();
        hasLevelGraph = buildLevelGraphBaseline(g, source, sink, levelEdges, &levelEdgeCount, dist, queue);
        if (stats != NULL) {
            stats->bfsCount++;
            stats->buildLevelGraphSeconds += tidalFlowSeqNowSeconds() - phaseStart;
        }
    }

    free(levelEdges);
    free(dist);
    free(queue);
    free(h);
    free(p);
    free(l);

    return maxFlow;
}

long long tidalFlowSeq_BLOCKING(Graph *g, int source, int sink) {
    return tidalFlowSeq_BLOCKING_STATS(g, source, sink, NULL);
}
