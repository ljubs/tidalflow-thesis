#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700

#include <errno.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "graph.h"
#include "tidalFlowSeqStats.h"

#define SEQ_BENCH_RUNS_ENV "SEQ_BENCH_RUNS"
#define SEQ_BENCH_TIMEOUT_ENV "SEQ_BENCH_TIMEOUT_SEC"
#define SEQ_BENCH_CSV_ENV "SEQ_BENCH_CSV"
#define SEQ_BENCH_RAW_CSV_ENV "SEQ_BENCH_RAW_CSV"

#define DEFAULT_RUNS 3
#define DEFAULT_TIMEOUT_SEC 1800

#define DEFAULT_DINIC_BIN "build/dinic"
#define DEFAULT_PUSH_RELABEL_BIN "build/pushRelabelHighest"
#define DEFAULT_DINITZ_BIN "build/dinitzBidirectional"

typedef struct {
    const char *name;
    const char *egrPath;
    const char *mode; /* auto | addReverseEdges | useExistingReverseEdges */
    int source;
    int sink;         /* -1 means last vertex */
} BenchmarkCase;

#include "benchmark_cases.h"

typedef struct {
    const char *label;
    const char *key;
    int isTidal;
    const char *defaultBin;
} SolverCase;

typedef struct {
    int ok;
    int timedOut;
    long long flow;
    double timeReadMs;
    double timeInitMs;
    double timeSolveMs;
    long long bfsCount;
    long long tideCycleCount;
    long long pruningCount;
    double buildLevelGraphSeconds;
    double pruningSeconds;
    double tideCycleSeconds;
    double highTideSeconds;
    double lowTideSeconds;
    double erosionSeconds;
    char output[65536];
} RunResult;

typedef struct {
    int nodes;
    int edges;
    int minDegree;
    int maxDegree;
    double avgDegree;
    int source;
    int sink;
} CaseMeta;

static const SolverCase solvers[] = {
    {"tidalFlowSeq_BASELINE", "baseline", 1, NULL},
    {"tidalFlowSeq_BLOCKING", "blocking", 1, NULL},
    {"tidalFlowSeq_PRUNING", "pruning", 1, NULL},
    {"tidalFlowSeq_COMBINED", "combined", 1, NULL},
    {"dinic", NULL, 0, DEFAULT_DINIC_BIN},
    {"dinitz-bidirectional", NULL, 0, DEFAULT_DINITZ_BIN},
    {"push-relabel-highest", NULL, 0, DEFAULT_PUSH_RELABEL_BIN},
};

static long long nowMillis(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((long long)ts.tv_sec * 1000LL) + (ts.tv_nsec / 1000000LL);
}

static int envInt(const char *name, int fallback) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') return fallback;

    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed <= 0 || parsed > 86400) {
        fprintf(stderr, "Invalid %s='%s', using %d\n", name, value, fallback);
        return fallback;
    }
    return (int)parsed;
}

static int envFlag(const char *name) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') return 0;
    return strcmp(value, "0") != 0 && strcmp(value, "false") != 0 && strcmp(value, "FALSE") != 0;
}

static int reverseModeFor(const char *mode) {
    if (strcmp(mode, "addReverseEdges") == 0) return ECL_REVERSE_ADD_RESIDUAL;
    if (strcmp(mode, "useExistingReverseEdges") == 0) return ECL_REVERSE_REQUIRE_EXISTING;
    return ECL_REVERSE_AUTO;
}

static int readEgrMetadata(const char *path, CaseMeta *metaOut) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "Could not open EGR file '%s': %s\n", path, strerror(errno));
        return -1;
    }

    int nodes = 0;
    int edges = 0;
    if (fread(&nodes, sizeof(int), 1, f) != 1 || fread(&edges, sizeof(int), 1, f) != 1) {
        fclose(f);
        fprintf(stderr, "Failed to read EGR header for '%s'\n", path);
        return -1;
    }

    if (nodes < 1 || edges < 0) {
        fclose(f);
        fprintf(stderr, "Invalid EGR header for '%s'\n", path);
        return -1;
    }

    int *nindex = (int *)malloc((size_t)(nodes + 1) * sizeof(int));
    if (nindex == NULL) {
        fclose(f);
        fprintf(stderr, "Could not allocate EGR index buffer for '%s'\n", path);
        return -1;
    }

    if (fread(nindex, sizeof(int), (size_t)(nodes + 1), f) != (size_t)(nodes + 1)) {
        free(nindex);
        fclose(f);
        fprintf(stderr, "Failed to read EGR index array for '%s'\n", path);
        return -1;
    }
    fclose(f);

    if (nindex[0] != 0 || nindex[nodes] != edges) {
        free(nindex);
        fprintf(stderr, "Invalid EGR index bounds for '%s'\n", path);
        return -1;
    }

    int minDegree = INT_MAX;
    int maxDegree = 0;
    for (int i = 0; i < nodes; i++) {
        if (nindex[i + 1] < nindex[i]) {
            free(nindex);
            fprintf(stderr, "Non-monotone EGR index array for '%s'\n", path);
            return -1;
        }
        int degree = nindex[i + 1] - nindex[i];
        if (degree < minDegree) minDegree = degree;
        if (degree > maxDegree) maxDegree = degree;
    }
    free(nindex);

    metaOut->nodes = nodes;
    metaOut->edges = edges;
    metaOut->minDegree = minDegree;
    metaOut->maxDegree = maxDegree;
    metaOut->avgDegree = (nodes > 0) ? ((double)edges / (double)nodes) : 0.0;
    return 0;
}

static int prepareCase(const BenchmarkCase *c, CaseMeta *metaOut) {
    if (readEgrMetadata(c->egrPath, metaOut) != 0) return -1;

    int source = c->source;
    int sink = (c->sink < 0) ? metaOut->nodes - 1 : c->sink;
    if (source < 0 || source >= metaOut->nodes || sink < 0 || sink >= metaOut->nodes || source == sink) {
        fprintf(stderr, "Invalid source/sink for %s (source=%d sink=%d nodes=%d)\n",
                c->name,
                source,
                sink,
                metaOut->nodes);
        return -1;
    }

    metaOut->source = source;
    metaOut->sink = sink;
    return 0;
}

static int parseMetricLine(const char *line, const char *prefix, double *out) {
    size_t n = strlen(prefix);
    if (strncmp(line, prefix, n) != 0) return 0;

    const char *p = line + n;
    while (*p == '\t' || *p == ' ' || *p == ':') p++;
    return sscanf(p, "%lf", out) == 1;
}

static int parseLongLongMetricLine(const char *line, const char *prefix, long long *out) {
    size_t n = strlen(prefix);
    if (strncmp(line, prefix, n) != 0) return 0;

    const char *p = line + n;
    while (*p == '\t' || *p == ' ' || *p == ':') p++;
    return sscanf(p, "%lld", out) == 1;
}

static void parseRunOutput(RunResult *result) {
    char *save = NULL;
    char *line = strtok_r(result->output, "\n", &save);
    while (line != NULL) {
        long long flow = 0;
        if (sscanf(line, "flow:%lld", &flow) == 1 ||
            sscanf(line, "flow:\t%lld", &flow) == 1 ||
            sscanf(line, "flow:\t\t%lld", &flow) == 1) {
            result->flow = flow;
        }

        parseMetricLine(line, "time read", &result->timeReadMs);
        parseMetricLine(line, "time init", &result->timeInitMs);
        parseMetricLine(line, "time solve", &result->timeSolveMs);
        parseLongLongMetricLine(line, "tidal bfs count", &result->bfsCount);
        parseLongLongMetricLine(line, "tidal tide cycle count", &result->tideCycleCount);
        parseLongLongMetricLine(line, "tidal pruning count", &result->pruningCount);
        parseMetricLine(line, "tidal build level graph", &result->buildLevelGraphSeconds);
        parseMetricLine(line, "tidal pruning", &result->pruningSeconds);
        parseMetricLine(line, "tidal tide cycle", &result->tideCycleSeconds);
        parseMetricLine(line, "tidal high tide", &result->highTideSeconds);
        parseMetricLine(line, "tidal low tide", &result->lowTideSeconds);
        parseMetricLine(line, "tidal erosion", &result->erosionSeconds);
        line = strtok_r(NULL, "\n", &save);
    }
}

static int runProgram(char *const argv[], int timeoutSec, RunResult *result) {
    memset(result, 0, sizeof(*result));

    int pipeFd[2];
    if (pipe(pipeFd) != 0) {
        perror("pipe");
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        close(pipeFd[0]);
        close(pipeFd[1]);
        return -1;
    }

    if (pid == 0) {
        close(pipeFd[0]);
        dup2(pipeFd[1], STDOUT_FILENO);
        dup2(pipeFd[1], STDERR_FILENO);
        close(pipeFd[1]);
        execv(argv[0], argv);
        fprintf(stderr, "exec failed for %s: %s\n", argv[0], strerror(errno));
        _exit(127);
    }

    close(pipeFd[1]);

    long long start = nowMillis();
    size_t used = 0;
    int status = 0;
    int childDone = 0;

    while (!childDone) {
        struct pollfd pfd;
        pfd.fd = pipeFd[0];
        pfd.events = POLLIN;
        pfd.revents = 0;

        int pollRc = poll(&pfd, 1, 50);
        if (pollRc > 0 && (pfd.revents & POLLIN)) {
            ssize_t got = read(pipeFd[0],
                               result->output + used,
                               sizeof(result->output) - used - 1);
            if (got > 0) used += (size_t)got;
        }

        pid_t waitRc = waitpid(pid, &status, WNOHANG);
        if (waitRc == pid) {
            childDone = 1;
            break;
        }

        long long elapsed = nowMillis() - start;
        if (elapsed > ((long long)timeoutSec * 1000LL)) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            result->timedOut = 1;
            childDone = 1;
        }
    }

    for (;;) {
        ssize_t got = read(pipeFd[0],
                           result->output + used,
                           sizeof(result->output) - used - 1);
        if (got <= 0) break;
        used += (size_t)got;
        if (used >= sizeof(result->output) - 1) break;
    }

    close(pipeFd[0]);
    result->output[used] = '\0';

    if (!result->timedOut && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        result->ok = 1;
        parseRunOutput(result);
    }

    return result->ok ? 0 : -1;
}

static long long runTidalVariant(const char *variant, Graph *g, int source, int sink, TidalFlowSeqStats *stats) {
    if (strcmp(variant, "baseline") == 0) return tidalFlowSeq_BASELINE_STATS(g, source, sink, stats);
    if (strcmp(variant, "blocking") == 0) return tidalFlowSeq_BLOCKING_STATS(g, source, sink, stats);
    if (strcmp(variant, "pruning") == 0) return tidalFlowSeq_PRUNING_STATS(g, source, sink, stats);
    if (strcmp(variant, "combined") == 0) return tidalFlowSeq_COMBINED_STATS(g, source, sink, stats);

    fprintf(stderr, "Unknown Tidal Flow variant: %s\n", variant);
    return -1;
}

static int runTidalChild(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "USAGE: %s --run-tidal <variant> <graph.egr> <source> <sink> <mode>\n", argv[0]);
        return 1;
    }

    const char *variant = argv[2];
    const char *graphPath = argv[3];
    int source = atoi(argv[4]);
    int sink = atoi(argv[5]);
    const char *mode = argv[6];

    Graph graph;
    long long readStart = nowMillis();
    if (buildGraphFromECLFileWithMode(&graph, graphPath, source, reverseModeFor(mode)) != 0) {
        return 1;
    }
    long long readEnd = nowMillis();

    long long solveStart = nowMillis();
    TidalFlowSeqStats stats;
    long long flow = runTidalVariant(variant, &graph, source, sink, &stats);
    long long solveEnd = nowMillis();
    freeGraph(&graph);

    if (flow < 0) return 1;

    printf("solver:\t\ttidalFlowSeq_%s\n", variant);
    printf("filename:\t%s\n", graphPath);
    printf("flow:\t\t%lld\n", flow);
    printf("time read:\t%lld ms\n", readEnd - readStart);
    printf("time init:\t0 ms\n");
    printf("time solve:\t%lld ms\n", solveEnd - solveStart);
    printf("# of threads:\t1\n");
    printf("tidal bfs count:\t%lld\n", stats.bfsCount);
    printf("tidal tide cycle count:\t%lld\n", stats.tideCycleCount);
    printf("tidal pruning count:\t%lld\n", stats.pruningCount);
    printf("tidal build level graph:\t%.9f\n", stats.buildLevelGraphSeconds);
    printf("tidal pruning:\t%.9f\n", stats.pruningSeconds);
    printf("tidal tide cycle:\t%.9f\n", stats.tideCycleSeconds);
    printf("tidal high tide:\t%.9f\n", stats.highTideSeconds);
    printf("tidal low tide:\t%.9f\n", stats.lowTideSeconds);
    printf("tidal erosion:\t%.9f\n", stats.erosionSeconds);
    return 0;
}

static void printRawCsvHeader(void) {
    printf("case,path,mode,nodes,edges,min_degree,avg_degree,max_degree,source,sink,solver,run,status,max_flow,time_read_s,time_init_s,time_solve_s,bfs_count,tide_cycle_count,pruning_count,build_level_graph_s,pruning_s,tide_cycle_s,high_tide_s,low_tide_s,erosion_s\n");
}

static void printRawCsvRow(const BenchmarkCase *benchmarkCase, const CaseMeta *meta, const char *solver,
                        int run, const char *status, const RunResult *result) {
    printf("%s,%s,%s,%d,%d,%d,%.6f,%d,%d,%d,%s,%d,%s,%lld,%.6f,%.6f,%.6f,%lld,%lld,%lld,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
           benchmarkCase->name,
           benchmarkCase->egrPath,
           benchmarkCase->mode,
           meta->nodes,
           meta->edges,
           meta->minDegree,
           meta->avgDegree,
           meta->maxDegree,
           meta->source,
           meta->sink,
           solver,
           run,
           status,
           result->flow,
           result->timeReadMs / 1000.0,
           result->timeInitMs / 1000.0,
           result->timeSolveMs / 1000.0,
           result->bfsCount,
           result->tideCycleCount,
           result->pruningCount,
           result->buildLevelGraphSeconds,
           result->pruningSeconds,
           result->tideCycleSeconds,
           result->highTideSeconds,
           result->lowTideSeconds,
           result->erosionSeconds);
    fflush(stdout);
}

static int compareDouble(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double medianOf(double *values, int count) {
    qsort(values, (size_t)count, sizeof(double), compareDouble);
    if ((count % 2) == 1) return values[count / 2];
    return (values[(count / 2) - 1] + values[count / 2]) / 2.0;
}

static void printSummaryCsvHeader(void) {
    printf("case,path,mode,nodes,edges,min_degree,avg_degree,max_degree,source,sink,solver,ok_runs,total_runs,max_flow,flow_consistent,median_s,avg_s,min_s,max_s,avg_bfs_count,avg_tide_cycle_count,avg_pruning_count,avg_build_level_graph_s,avg_pruning_s,avg_tide_cycle_s,avg_high_tide_s,avg_low_tide_s,avg_erosion_s,status\n");
}

static void printSummaryCsvRow(const BenchmarkCase *benchmarkCase, const CaseMeta *meta,
                               const char *solver, int okRuns, int totalRuns, long long flow,
                               int flowConsistent, double medianMs, double avgMs,
                               double minMs, double maxMs, double avgBfsCount,
                               double avgTideCycleCount, double avgPruningCount,
                               double avgBuildLevelGraphSeconds, double avgPruningSeconds,
                               double avgTideCycleSeconds, double avgHighTideSeconds,
                               double avgLowTideSeconds, double avgErosionSeconds,
                               const char *status) {
    printf("%s,%s,%s,%d,%d,%d,%.6f,%d,%d,%d,%s,%d,%d,%lld,%s,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%s\n",
           benchmarkCase->name,
           benchmarkCase->egrPath,
           benchmarkCase->mode,
           meta->nodes,
           meta->edges,
           meta->minDegree,
           meta->avgDegree,
           meta->maxDegree,
           meta->source,
           meta->sink,
           solver,
           okRuns,
           totalRuns,
           flow,
           flowConsistent ? "yes" : "no",
           medianMs / 1000.0,
           avgMs / 1000.0,
           minMs / 1000.0,
           maxMs / 1000.0,
           avgBfsCount,
           avgTideCycleCount,
           avgPruningCount,
           avgBuildLevelGraphSeconds,
           avgPruningSeconds,
           avgTideCycleSeconds,
           avgHighTideSeconds,
           avgLowTideSeconds,
           avgErosionSeconds,
           status);
    fflush(stdout);
}

static void printSummaryHeader(void) {
    printf("%-22s %-24s %-10s %-10s %-7s %-9s %-7s %-8s %-8s %-26s %7s %14s %-15s %13s %13s %13s %13s %8s %9s %8s %10s %9s %10s %9s %9s %9s %s\n",
           "case",
           "mode",
           "nodes",
           "edges",
           "minDeg",
           "avgDeg",
           "maxDeg",
           "source",
           "sink",
           "solver",
           "ok/runs",
           "max_flow",
           "flow_consistent",
           "median_s",
           "avg_s",
           "min_s",
           "max_s",
           "bfs",
           "tideCyc",
           "prunes",
           "build_s",
           "prune_s",
           "tide_s",
           "high_s",
           "low_s",
           "erosion_s",
           "status");
}

static void printSummaryRow(const BenchmarkCase *benchmarkCase, const CaseMeta *meta, const char *solver,
                            int okRuns, int totalRuns, long long flow,
                            int flowConsistent,
                            double medianMs, double avgMs, double minMs, double maxMs,
                            double avgBfsCount, double avgTideCycleCount, double avgPruningCount,
                            double avgBuildLevelGraphSeconds, double avgPruningSeconds,
                            double avgTideCycleSeconds, double avgHighTideSeconds,
                            double avgLowTideSeconds, double avgErosionSeconds,
                            const char *status) {
    char runText[32];
    snprintf(runText, sizeof(runText), "%d/%d", okRuns, totalRuns);
    printf("%-22s %-24s %-10d %-10d %-7d %-9.2f %-7d %-8d %-8d %-26s %7s %14lld %-15s %13.6f %13.6f %13.6f %13.6f %8.1f %9.1f %8.1f %10.6f %9.6f %10.6f %9.6f %9.6f %9.6f %s\n",
           benchmarkCase->name,
           benchmarkCase->mode,
           meta->nodes,
           meta->edges,
           meta->minDegree,
           meta->avgDegree,
           meta->maxDegree,
           meta->source,
           meta->sink,
           solver,
           runText,
           flow,
           flowConsistent ? "yes" : "no",
           medianMs / 1000.0,
           avgMs / 1000.0,
           minMs / 1000.0,
           maxMs / 1000.0,
           avgBfsCount,
           avgTideCycleCount,
           avgPruningCount,
           avgBuildLevelGraphSeconds,
           avgPruningSeconds,
           avgTideCycleSeconds,
           avgHighTideSeconds,
           avgLowTideSeconds,
           avgErosionSeconds,
           status);
    fflush(stdout);
}

static void printRunInfo(int runs, int timeoutSec, int summaryCsv, int rawCsv) {
    char host[256] = "unknown";
    if (gethostname(host, sizeof(host)) != 0) {
        snprintf(host, sizeof(host), "unknown");
    }
    host[sizeof(host) - 1] = '\0';

    time_t now = time(NULL);
    char timeText[64] = "unknown";
    struct tm tmValue;
    if (now != (time_t)-1 && localtime_r(&now, &tmValue) != NULL) {
        strftime(timeText, sizeof(timeText), "%Y-%m-%d %H:%M:%S %Z", &tmValue);
    }

    fprintf(stderr, "Sequential benchmark configuration\n");
    fprintf(stderr, "- run timestamp: %s\n", timeText);
    fprintf(stderr, "- hostname: %s\n", host);
    fprintf(stderr, "- graph cases: %zu\n", sizeof(benchmarkCases) / sizeof(benchmarkCases[0]));
    fprintf(stderr, "- solvers: %zu\n", sizeof(solvers) / sizeof(solvers[0]));
    fprintf(stderr, "- runs per solver/graph: %d (%s)\n", runs, SEQ_BENCH_RUNS_ENV);
    fprintf(stderr, "- default timeout per run: %d seconds (%s)\n", timeoutSec, SEQ_BENCH_TIMEOUT_ENV);
    fprintf(stderr, "- output mode: %s\n", rawCsv ? "raw csv" : (summaryCsv ? "summary csv" : "summary table"));
    fprintf(stderr, "- timing: seconds, solve time only in summary; read/init excluded\n");
    fprintf(stderr, "- benchmark build: %s %s\n", __DATE__, __TIME__);
#ifdef __VERSION__
    fprintf(stderr, "- C compiler: %s\n", __VERSION__);
#endif
    fprintf(stderr, "- default binaries: %s, %s, %s\n",
            DEFAULT_DINIC_BIN,
            DEFAULT_DINITZ_BIN,
            DEFAULT_PUSH_RELABEL_BIN);
}

static int runBenchmark(const char *selfPath) {
    int failures = 0;
    int runs = envInt(SEQ_BENCH_RUNS_ENV, DEFAULT_RUNS);
    int timeoutSec = envInt(SEQ_BENCH_TIMEOUT_ENV, DEFAULT_TIMEOUT_SEC);
    int summaryCsv = envFlag(SEQ_BENCH_CSV_ENV);
    int rawCsv = envFlag(SEQ_BENCH_RAW_CSV_ENV);

    printRunInfo(runs, timeoutSec, summaryCsv, rawCsv);

    if (rawCsv) printRawCsvHeader();
    else if (summaryCsv) printSummaryCsvHeader();
    else printSummaryHeader();

    for (size_t c = 0; c < sizeof(benchmarkCases) / sizeof(benchmarkCases[0]); c++) {
        CaseMeta meta;
        if (prepareCase(&benchmarkCases[c], &meta) != 0) {
            failures++;
            continue;
        }
        fprintf(stderr,
                "- case %s source=%d sink=%d timeout=%d seconds\n",
                benchmarkCases[c].name,
                meta.source,
                meta.sink,
                timeoutSec);

        long long expectedFlow = -1;
        for (size_t s = 0; s < sizeof(solvers) / sizeof(solvers[0]); s++) {
            double *solveTimes = (double *)calloc((size_t)runs, sizeof(double));
            if (solveTimes == NULL) {
                fprintf(stderr, "Could not allocate timing buffer\n");
                return 1;
            }
            int okRuns = 0;
            int solverFailed = 0;
            long long solverFlow = -1;
            double sumSolve = 0.0;
            double minSolve = 0.0;
            double maxSolve = 0.0;
            double sumBfsCount = 0.0;
            double sumTideCycleCount = 0.0;
            double sumPruningCount = 0.0;
            double sumBuildLevelGraphSeconds = 0.0;
            double sumPruningSeconds = 0.0;
            double sumTideCycleSeconds = 0.0;
            double sumHighTideSeconds = 0.0;
            double sumLowTideSeconds = 0.0;
            double sumErosionSeconds = 0.0;
            const char *solverStatus = "ok";

            for (int run = 1; run <= runs; run++) {
                char sourceText[32];
                char sinkText[32];
                snprintf(sourceText, sizeof(sourceText), "%d", meta.source);
                snprintf(sinkText, sizeof(sinkText), "%d", meta.sink);

                char *tidalArgv[] = {
                    (char *)selfPath,
                    (char *)"--run-tidal",
                    (char *)solvers[s].key,
                    (char *)benchmarkCases[c].egrPath,
                    sourceText,
                    sinkText,
                    (char *)benchmarkCases[c].mode,
                    NULL,
                };

                const char *bin = solvers[s].isTidal ? selfPath : solvers[s].defaultBin;
                char *externalArgv[] = {
                    (char *)bin,
                    (char *)benchmarkCases[c].egrPath,
                    sourceText,
                    sinkText,
                    (char *)benchmarkCases[c].mode,
                    NULL,
                };

                RunResult result;
                char *const *argv = solvers[s].isTidal ? tidalArgv : externalArgv;
                int ok = (runProgram((char *const *)argv, timeoutSec, &result) == 0);

                const char *status = "ok";
                if (!ok) {
                    status = result.timedOut ? "timeout" : "error";
                    solverStatus = status;
                    solverFailed = 1;
                    failures++;
                } else if (expectedFlow < 0) {
                    expectedFlow = result.flow;
                } else if (result.flow != expectedFlow) {
                    status = "flow-mismatch";
                    solverStatus = status;
                    solverFailed = 1;
                    failures++;
                }

                if (ok && strcmp(status, "ok") == 0) {
                    if (solverFlow < 0) solverFlow = result.flow;
                    solveTimes[okRuns++] = result.timeSolveMs;
                    sumSolve += result.timeSolveMs;
                    if (okRuns == 1 || result.timeSolveMs < minSolve) minSolve = result.timeSolveMs;
                    if (okRuns == 1 || result.timeSolveMs > maxSolve) maxSolve = result.timeSolveMs;
                    sumBfsCount += (double)result.bfsCount;
                    sumTideCycleCount += (double)result.tideCycleCount;
                    sumPruningCount += (double)result.pruningCount;
                    sumBuildLevelGraphSeconds += result.buildLevelGraphSeconds;
                    sumPruningSeconds += result.pruningSeconds;
                    sumTideCycleSeconds += result.tideCycleSeconds;
                    sumHighTideSeconds += result.highTideSeconds;
                    sumLowTideSeconds += result.lowTideSeconds;
                    sumErosionSeconds += result.erosionSeconds;
                }

                if (rawCsv) {
                    printRawCsvRow(&benchmarkCases[c], &meta, solvers[s].label, run, status, &result);
                }

                if (!ok && result.output[0] != '\0') {
                    fprintf(stderr, "Output from failed run (%s / %s):\n%s\n",
                            benchmarkCases[c].name,
                            solvers[s].label,
                            result.output);
                }
            }

            if (!rawCsv) {
                double median = 0.0;
                double avg = 0.0;
                if (okRuns > 0) {
                    median = medianOf(solveTimes, okRuns);
                    avg = sumSolve / okRuns;
                }
                double avgBfsCount = okRuns > 0 ? sumBfsCount / okRuns : 0.0;
                double avgTideCycleCount = okRuns > 0 ? sumTideCycleCount / okRuns : 0.0;
                double avgPruningCount = okRuns > 0 ? sumPruningCount / okRuns : 0.0;
                double avgBuildLevelGraphSeconds = okRuns > 0 ? sumBuildLevelGraphSeconds / okRuns : 0.0;
                double avgPruningSeconds = okRuns > 0 ? sumPruningSeconds / okRuns : 0.0;
                double avgTideCycleSeconds = okRuns > 0 ? sumTideCycleSeconds / okRuns : 0.0;
                double avgHighTideSeconds = okRuns > 0 ? sumHighTideSeconds / okRuns : 0.0;
                double avgLowTideSeconds = okRuns > 0 ? sumLowTideSeconds / okRuns : 0.0;
                double avgErosionSeconds = okRuns > 0 ? sumErosionSeconds / okRuns : 0.0;
                if (solverFailed == 0 && okRuns != runs) solverStatus = "partial";
                int flowConsistent = (okRuns > 0 && expectedFlow >= 0 && solverFlow == expectedFlow);
                if (summaryCsv) {
                    printSummaryCsvRow(&benchmarkCases[c],
                                       &meta,
                                       solvers[s].label,
                                       okRuns,
                                       runs,
                                       solverFlow,
                                       flowConsistent,
                                       median,
                                       avg,
                                       minSolve,
                                       maxSolve,
                                       avgBfsCount,
                                       avgTideCycleCount,
                                       avgPruningCount,
                                       avgBuildLevelGraphSeconds,
                                       avgPruningSeconds,
                                       avgTideCycleSeconds,
                                       avgHighTideSeconds,
                                       avgLowTideSeconds,
                                       avgErosionSeconds,
                                       solverStatus);
                } else {
                    printSummaryRow(&benchmarkCases[c],
                                    &meta,
                                    solvers[s].label,
                                    okRuns,
                                    runs,
                                    solverFlow,
                                    flowConsistent,
                                    median,
                                    avg,
                                    minSolve,
                                    maxSolve,
                                    avgBfsCount,
                                    avgTideCycleCount,
                                    avgPruningCount,
                                    avgBuildLevelGraphSeconds,
                                    avgPruningSeconds,
                                    avgTideCycleSeconds,
                                    avgHighTideSeconds,
                                    avgLowTideSeconds,
                                    avgErosionSeconds,
                                    solverStatus);
                }
            }

            free(solveTimes);
        }
    }

    if (failures != 0) {
        fflush(stdout);
        printf("Sequential benchmark finished with %d failed run(s).\n", failures);
        return 1;
    }

    fflush(stdout);
    printf("Sequential benchmark finished successfully.\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--run-tidal") == 0) {
        return runTidalChild(argc, argv);
    }

    return runBenchmark(argv[0]);
}
