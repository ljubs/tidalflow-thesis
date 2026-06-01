#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graph.h"

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

void initGraph(Graph *g, int n, int m) {
    g->nodeCount = n;
    g->edgeCount = m;

    g->ver = (int *)malloc((size_t)(n + 1) * sizeof(int));
    g->src = (m > 0) ? (int *)malloc((size_t)m * sizeof(int)) : NULL;
    g->dst = (m > 0) ? (int *)malloc((size_t)m * sizeof(int)) : NULL;
    g->rev = (m > 0) ? (int *)malloc((size_t)m * sizeof(int)) : NULL;
    g->capacity = (m > 0) ? (int *)malloc((size_t)m * sizeof(int)) : NULL;
    g->flow = (m > 0) ? (int *)calloc((size_t)m, sizeof(int)) : NULL;

    if ((g->ver == NULL) ||
        ((m > 0) && ((g->src == NULL) || (g->dst == NULL) || (g->rev == NULL) ||
                     (g->capacity == NULL) || (g->flow == NULL)))) {
        die("Graph allocation failed");
    }
}

/*
 * NOT USED
 * for undirected edge-list inputs on stdin:
 * n m
 * u v c  (1-indexed nodes)
 */
void buildGraphFromStdin(Graph *g) {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) {
        die("Invalid input header");
    }
    initGraph(g, n, 2 * m);
    Edge *edgeList = (Edge *)malloc((size_t)m * sizeof(Edge));
    if (edgeList == NULL) die("Edge list allocation failed");

    for (int i = 0; i < m; i++) {
        if (scanf("%d %d %d", &edgeList[i].u, &edgeList[i].v, &edgeList[i].c) != 3) {
            die("Invalid edge row");
        }
        edgeList[i].u--;
        edgeList[i].v--;
        if ((edgeList[i].u < 0) || (edgeList[i].u >= n) ||
            (edgeList[i].v < 0) || (edgeList[i].v >= n)) {
            die("Edge endpoint out of range");
        }
    }

    int *degree = (int *)calloc((size_t)g->nodeCount, sizeof(int));
    if (degree == NULL) die("Degree allocation failed");
    for (int i = 0; i < m; i++) {
        degree[edgeList[i].u]++;
        degree[edgeList[i].v]++;
    }

    g->ver[0] = 0;
    for (int i = 1; i <= n; i++) {
        g->ver[i] = g->ver[i - 1] + degree[i - 1];
    }

    int *nextPosition = (int *)calloc((size_t)n, sizeof(int));
    if (nextPosition == NULL) die("Position allocation failed");
    for (int i = 0; i < m; i++) {
        int u = edgeList[i].u;
        int v = edgeList[i].v;
        int c = edgeList[i].c;

        int idxU = g->ver[u] + nextPosition[u]++;
        g->src[idxU] = u;
        g->dst[idxU] = v;
        g->capacity[idxU] = c;

        int idxV = g->ver[v] + nextPosition[v]++;
        g->src[idxV] = v;
        g->dst[idxV] = u;
        g->capacity[idxV] = c;

        g->rev[idxU] = idxV;
        g->rev[idxV] = idxU;
    }

    free(edgeList);
    free(degree);
    free(nextPosition);
}

typedef struct {
    unsigned long long key;
    int head;
} PairBucket;

static unsigned long long packPairKey(int u, int v) {
    return (((unsigned long long)(unsigned int)u) << 32) | (unsigned int)v;
}

static size_t hashKey(unsigned long long x) {
    /* splitmix64 finalizer */
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return (size_t)x;
}

static size_t nextPow2(size_t x) {
    size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

static size_t findBucket(const PairBucket *table, size_t mask, unsigned long long key) {
    size_t slot = hashKey(key) & mask;
    while ((table[slot].key != ULLONG_MAX) && (table[slot].key != key)) {
        slot = (slot + 1) & mask;
    }
    return slot;
}

static int buildReverseEdgesByPairing(Graph *g, const char *path, int reportMissing) {
    const int m = g->edgeCount;
    const size_t tableSize = nextPow2((size_t)m + 1);
    const size_t mask = tableSize - 1;

    PairBucket *table = (PairBucket *)malloc(tableSize * sizeof(PairBucket));
    int *nextEdge = (int *)malloc((size_t)m * sizeof(int));
    if ((table == NULL) || (nextEdge == NULL)) {
        free(table);
        free(nextEdge);
        die("Reverse-edge map allocation failed");
    }

    for (size_t i = 0; i < tableSize; i++) {
        table[i].key = ULLONG_MAX;
        table[i].head = -1;
    }

    for (int e = 0; e < m; e++) {
        unsigned long long key = packPairKey(g->src[e], g->dst[e]);
        size_t slot = findBucket(table, mask, key);
        if (table[slot].key == ULLONG_MAX) table[slot].key = key;
        nextEdge[e] = table[slot].head;
        table[slot].head = e;
    }

    for (int e = 0; e < m; e++) g->rev[e] = -1;

    for (int e = 0; e < m; e++) {
        if (g->rev[e] != -1) continue;

        const int u = g->src[e];
        const int v = g->dst[e];
        const unsigned long long revKey = packPairKey(v, u);
        const size_t slot = findBucket(table, mask, revKey);
        if (table[slot].key != revKey || table[slot].head < 0) {
            if (reportMissing) {
                fprintf(stderr, "Missing reverse edge for (%d -> %d) in %s.\n", u, v, path);
            }
            free(table);
            free(nextEdge);
            return -1;
        }

        int rev = table[slot].head;
        while ((rev >= 0) && ((rev == e) || (g->rev[rev] != -1))) {
            rev = nextEdge[rev];
        }
        if (rev < 0) {
            if (reportMissing) {
                fprintf(stderr, "Missing reverse edge for (%d -> %d) in %s.\n", u, v, path);
            }
            free(table);
            free(nextEdge);
            return -1;
        }

        g->rev[e] = rev;
        g->rev[rev] = e;
    }

    free(table);
    free(nextEdge);
    return 0;
}

typedef struct {
    int n;
    int m;
    int *ver;
    int *edges;
    int *capacity;
} RawECL;

static void freeRawECL(RawECL *raw) {
    free(raw->ver);
    free(raw->edges);
    free(raw->capacity);
    raw->ver = NULL;
    raw->edges = NULL;
    raw->capacity = NULL;
    raw->n = 0;
    raw->m = 0;
}

static int readRawECL(const char *path, int source, RawECL *raw) {
    memset(raw, 0, sizeof(*raw));

    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "Could not open ECL file: %s\n", path);
        return -1;
    }

    int n = 0, m = 0;
    if (fread(&n, sizeof(int), 1, f) != 1 || fread(&m, sizeof(int), 1, f) != 1) {
        fclose(f);
        fprintf(stderr, "Failed to read ECL header: %s\n", path);
        return -1;
    }
    if ((n < 1) || (m < 0)) {
        fclose(f);
        fprintf(stderr, "Invalid ECL node/edge counts: %s\n", path);
        return -1;
    }

    raw->n = n;
    raw->m = m;
    raw->ver = (int *)malloc((size_t)(n + 1) * sizeof(int));
    raw->edges = (m > 0) ? (int *)malloc((size_t)m * sizeof(int)) : NULL;
    raw->capacity = (m > 0) ? (int *)malloc((size_t)m * sizeof(int)) : NULL;
    if ((raw->ver == NULL) || ((m > 0) && ((raw->edges == NULL) || (raw->capacity == NULL)))) {
        fclose(f);
        freeRawECL(raw);
        die("Raw ECL allocation failed");
    }

    if (fread(raw->ver, sizeof(int), (size_t)(n + 1), f) != (size_t)(n + 1)) {
        fclose(f);
        fprintf(stderr, "Failed to read ECL nindex: %s\n", path);
        freeRawECL(raw);
        return -1;
    }
    if ((m > 0) && (fread(raw->edges, sizeof(int), (size_t)m, f) != (size_t)m)) {
        fclose(f);
        fprintf(stderr, "Failed to read ECL nlist: %s\n", path);
        freeRawECL(raw);
        return -1;
    }

    if (m > 0) {
        int *tmpWeights = (int *)malloc((size_t)m * sizeof(int));
        if (tmpWeights == NULL) {
            fclose(f);
            freeRawECL(raw);
            die("Weight buffer allocation failed");
        }

        size_t gotWeights = fread(tmpWeights, sizeof(int), (size_t)m, f);
        if (gotWeights == (size_t)m) {
            for (int e = 0; e < m; e++) raw->capacity[e] = abs(tmpWeights[e]);
        } else if (gotWeights == 0) {
            srand(source);
            for (int e = 0; e < m; e++) raw->capacity[e] = rand() % n;
        } else {
            free(tmpWeights);
            fclose(f);
            fprintf(stderr, "Corrupt ECL weight section: %s\n", path);
            freeRawECL(raw);
            return -1;
        }
        free(tmpWeights);
    }

    fclose(f);

    if (raw->ver[0] != 0 || raw->ver[n] != m) {
        fprintf(stderr, "Invalid ECL index bounds: %s\n", path);
        freeRawECL(raw);
        return -1;
    }
    for (int i = 1; i <= n; i++) {
        if (raw->ver[i] < raw->ver[i - 1]) {
            fprintf(stderr, "ECL index array is not monotonic: %s\n", path);
            freeRawECL(raw);
            return -1;
        }
    }
    for (int u = 0; u < n; u++) {
        for (int e = raw->ver[u]; e < raw->ver[u + 1]; e++) {
            int v = raw->edges[e];
            if ((v < 0) || (v >= n)) {
                fprintf(stderr, "ECL edge endpoint out of range in %s\n", path);
                freeRawECL(raw);
                return -1;
            }
        }
    }
    return 0;
}

static int buildGraphFromRawStrict(Graph *g, const RawECL *raw, const char *path, int reportMissing) {
    initGraph(g, raw->n, raw->m);
    memcpy(g->ver, raw->ver, (size_t)(raw->n + 1) * sizeof(int));
    if (raw->m > 0) {
        memcpy(g->dst, raw->edges, (size_t)raw->m * sizeof(int));
        memcpy(g->capacity, raw->capacity, (size_t)raw->m * sizeof(int));
    }
    for (int u = 0; u < raw->n; u++) {
        for (int e = g->ver[u]; e < g->ver[u + 1]; e++) {
            g->src[e] = u;
        }
    }
    if (buildReverseEdgesByPairing(g, path, reportMissing) != 0) {
        freeGraph(g);
        return -1;
    }
    return 0;
}

static int buildGraphFromRawResidual(Graph *g, const RawECL *raw) {
    if (raw->m > (INT_MAX / 2)) {
        fprintf(stderr, "Graph has too many edges to add residual reverse arcs\n");
        return -1;
    }

    const int n = raw->n;
    const int m = raw->m;
    const int mr = 2 * m;

    initGraph(g, n, mr);

    int *degree = (int *)calloc((size_t)n, sizeof(int));
    int *nextPos = (int *)calloc((size_t)n, sizeof(int));
    if ((degree == NULL) || (nextPos == NULL)) {
        free(degree);
        free(nextPos);
        freeGraph(g);
        die("Residual build allocation failed");
    }

    for (int u = 0; u < n; u++) {
        for (int e = raw->ver[u]; e < raw->ver[u + 1]; e++) {
            int v = raw->edges[e];
            degree[u]++;
            degree[v]++;
        }
    }

    g->ver[0] = 0;
    for (int i = 1; i <= n; i++) {
        g->ver[i] = g->ver[i - 1] + degree[i - 1];
    }

    for (int u = 0; u < n; u++) {
        for (int e = raw->ver[u]; e < raw->ver[u + 1]; e++) {
            int v = raw->edges[e];
            int c = raw->capacity[e];

            int fwd = g->ver[u] + nextPos[u]++;
            int rev = g->ver[v] + nextPos[v]++;

            g->src[fwd] = u;
            g->dst[fwd] = v;
            g->capacity[fwd] = c;
            g->rev[fwd] = rev;

            g->src[rev] = v;
            g->dst[rev] = u;
            g->capacity[rev] = 0;
            g->rev[rev] = fwd;
        }
    }

    free(degree);
    free(nextPos);
    return 0;
}

/*
* ECL binary CSR format:
* [int nodes][int edges][int nindex[nodes+1]][int nlist[edges]][optional int eweight[edges]]
*
* Capacity assignment intentionally mirrors ECL-MaxFlow:
* - if eweight exists: capacity[e] = abs(eweight[e])
* - else: srand(source), capacity[e] = rand() % nodes
*/
int buildGraphFromECLFileWithMode(Graph *g, const char *path, int source, int reverseMode) {
    memset(g, 0, sizeof(*g));

    RawECL raw;
    if (readRawECL(path, source, &raw) != 0) return -1;

    int rc = -1;
    if (reverseMode == ECL_REVERSE_REQUIRE_EXISTING) {
        rc = buildGraphFromRawStrict(g, &raw, path, 1);
    } else if (reverseMode == ECL_REVERSE_ADD_RESIDUAL) {
        rc = buildGraphFromRawResidual(g, &raw);
    } else if (reverseMode == ECL_REVERSE_AUTO) {
        rc = buildGraphFromRawStrict(g, &raw, path, 0);
        if (rc != 0) {
            rc = buildGraphFromRawResidual(g, &raw);
        }
    } else {
        fprintf(stderr, "Unknown reverse mode: %d\n", reverseMode);
    }

    freeRawECL(&raw);
    return rc;
}

int buildGraphFromECLFile(Graph *g, const char *path, int source) {
    return buildGraphFromECLFileWithMode(g, path, source, ECL_REVERSE_AUTO);
}

// NOT USED
void buildGraph(Graph *g) { buildGraphFromStdin(g); }

void freeGraph(Graph *g) {
    free(g->ver);
    free(g->src);
    free(g->dst);
    free(g->rev);
    free(g->capacity);
    free(g->flow);
    g->ver = NULL;
    g->src = NULL;
    g->dst = NULL;
    g->rev = NULL;
    g->capacity = NULL;
    g->flow = NULL;
    g->nodeCount = 0;
    g->edgeCount = 0;
}
