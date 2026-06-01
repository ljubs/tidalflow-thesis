#ifndef TIDAL_FLOW_SEQ_STATS_H
#define TIDAL_FLOW_SEQ_STATS_H

#include <string.h>
#include <time.h>

#include "graph.h"

typedef struct {
    long long bfsCount;
    long long tideCycleCount;
    long long pruningCount;

    double buildLevelGraphSeconds;
    double pruningSeconds;
    double tideCycleSeconds;
    double highTideSeconds;
    double lowTideSeconds;
    double erosionSeconds;
} TidalFlowSeqStats;

static inline void resetTidalFlowSeqStats(TidalFlowSeqStats *stats) {
    if (stats != NULL) {
        memset(stats, 0, sizeof(*stats));
    }
}

static inline double tidalFlowSeqNowSeconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ((double)ts.tv_nsec / 1000000000.0);
}

long long tidalFlowSeq_BASELINE_STATS(Graph *g, int source, int sink, TidalFlowSeqStats *stats);
long long tidalFlowSeq_BLOCKING_STATS(Graph *g, int source, int sink, TidalFlowSeqStats *stats);
long long tidalFlowSeq_PRUNING_STATS(Graph *g, int source, int sink, TidalFlowSeqStats *stats);
long long tidalFlowSeq_COMBINED_STATS(Graph *g, int source, int sink, TidalFlowSeqStats *stats);

#endif
