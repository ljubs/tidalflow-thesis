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

#define CUDA_BENCH_RUNS_ENV "CUDA_BENCH_RUNS"
#define CUDA_BENCH_TIMEOUT_ENV "CUDA_BENCH_TIMEOUT_SEC"
#define CUDA_BENCH_BLOCK_SIZES_ENV "CUDA_BENCH_BLOCK_SIZES"
#define CUDA_BENCH_CSV_ENV "CUDA_BENCH_CSV"
#define CUDA_BENCH_RAW_CSV_ENV "CUDA_BENCH_RAW_CSV"
#define CUDA_BENCH_CASE_LIMIT_ENV "CUDA_BENCH_CASE_LIMIT"
#define CUDA_TIDAL_BIN_ENV "CUDA_TIDAL_BIN"
#define CUDA_TIDAL_PREFIX_BIN_ENV "CUDA_TIDAL_PREFIX_BIN"
#define CUDA_ECL_MAXFLOW_BIN_ENV "CUDA_ECL_MAXFLOW_BIN"
#define CUDA_DINITZ_BIN_ENV "CUDA_DINITZ_BIN"

#define DEFAULT_RUNS 3
#define DEFAULT_TIMEOUT_SEC 1800
#define DEFAULT_BLOCK_SIZES "128,256,512,1024"
#define DEFAULT_TIDAL_BIN "build/tidalFlowCUDA"
#define DEFAULT_TIDAL_PREFIX_BIN "build/tidalFlowCUDAPrefix"
#define DEFAULT_ECL_MAXFLOW_BIN "build/eclMaxFlow"
#define DEFAULT_DINITZ_BIN "build/dinitzBidirectional"
#define MAX_BLOCK_OPTIONS 4

typedef struct {
    const char *name;
    const char *egrPath;
    const char *mode; /* auto | addReverseEdges | useExistingReverseEdges */
    int source;
    int sink;         /* -1 means last vertex */
} BenchmarkCase;

#include "benchmark_cases.h"

typedef enum {
    SOLVER_TIDAL_CUDA,
    SOLVER_TIDAL_CUDA_PREFIX,
    SOLVER_ECL_MAXFLOW,
    SOLVER_DINITZ,
    SOLVER_TIDAL_SEQ_PRUNING
} SolverKind;

typedef struct {
    const char *label;
    const char *type;
    SolverKind kind;
    int supportsBlockSizes;
    int fixedBlockSize;
    const char *binEnv;
    const char *defaultBin;
} SolverCase;

typedef struct {
    int ok;
    int timedOut;
    long long flow;
    double timeReadMs;
    double timeInitMs;
    double timeSolveMs;
    int blockSize;
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
    {"tidalFlowCUDA", "cuda", SOLVER_TIDAL_CUDA, 1, 0, CUDA_TIDAL_BIN_ENV, DEFAULT_TIDAL_BIN},
    {"tidalFlowCUDAPrefix", "cuda", SOLVER_TIDAL_CUDA_PREFIX, 1, 0, CUDA_TIDAL_PREFIX_BIN_ENV, DEFAULT_TIDAL_PREFIX_BIN},
    {"ECL-MaxFlow", "cuda", SOLVER_ECL_MAXFLOW, 0, 512, CUDA_ECL_MAXFLOW_BIN_ENV, DEFAULT_ECL_MAXFLOW_BIN},
    {"dinitz-bidirectional", "sequential", SOLVER_DINITZ, 0, 0, CUDA_DINITZ_BIN_ENV, DEFAULT_DINITZ_BIN},
    {"tidalFlowSeq_PRUNING", "sequential", SOLVER_TIDAL_SEQ_PRUNING, 0, 0, NULL, NULL},
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
    if (errno != 0 || end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        fprintf(stderr, "Invalid %s='%s', using %d\n", name, value, fallback);
        return fallback;
    }
    return (int)parsed;
}

static int envPositiveInt(const char *name, int fallback) {
    int value = envInt(name, fallback);
    return value > 0 ? value : fallback;
}

static int envFlag(const char *name) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') return 0;
    return strcmp(value, "0") != 0 && strcmp(value, "false") != 0 && strcmp(value, "FALSE") != 0;
}

static const char *envString(const char *name, const char *fallback) {
    if (name == NULL) return fallback;
    const char *value = getenv(name);
    return (value != NULL && value[0] != '\0') ? value : fallback;
}

static int reverseModeFor(const char *mode) {
    if (strcmp(mode, "addReverseEdges") == 0) return ECL_REVERSE_ADD_RESIDUAL;
    if (strcmp(mode, "useExistingReverseEdges") == 0) return ECL_REVERSE_REQUIRE_EXISTING;
    return ECL_REVERSE_AUTO;
}

static int isSupportedBlockSize(int blockSize) {
    return blockSize == 128 || blockSize == 256 || blockSize == 512 || blockSize == 1024;
}

static int appendBlockSize(int *blockSizes, int *count, int value) {
    if (!isSupportedBlockSize(value)) return -1;
    for (int i = 0; i < *count; i++) {
        if (blockSizes[i] == value) return 0;
    }
    if (*count >= MAX_BLOCK_OPTIONS) return -1;
    blockSizes[(*count)++] = value;
    return 0;
}

static int parseBlockSizeList(int *blockSizes, int *blockSizeCount) {
    *blockSizeCount = 0;
    const char *list = getenv(CUDA_BENCH_BLOCK_SIZES_ENV);
    if (list == NULL || list[0] == '\0') list = DEFAULT_BLOCK_SIZES;

    char buffer[1024];
    snprintf(buffer, sizeof(buffer), "%s", list);
    char *save = NULL;
    for (char *token = strtok_r(buffer, ",", &save);
         token != NULL;
         token = strtok_r(NULL, ",", &save)) {
        char *end = NULL;
        errno = 0;
        long parsed = strtol(token, &end, 10);
        if (errno != 0 || end == token || *end != '\0' ||
            parsed <= 0 || parsed > INT_MAX ||
            appendBlockSize(blockSizes, blockSizeCount, (int)parsed) != 0) {
            fprintf(stderr,
                    "Invalid %s='%s'. Supported values: 128,256,512,1024\n",
                    CUDA_BENCH_BLOCK_SIZES_ENV,
                    list);
            return -1;
        }
    }

    return *blockSizeCount > 0 ? 0 : -1;
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

static int parseIntMetricLine(const char *line, const char *prefix, int *out) {
    size_t n = strlen(prefix);
    if (strncmp(line, prefix, n) != 0) return 0;

    const char *p = line + n;
    while (*p == '\t' || *p == ' ' || *p == ':') p++;
    return sscanf(p, "%d", out) == 1;
}

static int parseEclMaxFlowLine(const char *line, long long *out) {
    const char *prefix = "maximum flow: ";
    size_t n = strlen(prefix);
    if (strncmp(line, prefix, n) == 0) {
        const char *p = line + n;
        while (*p == '\t' || *p == ' ') p++;
        if (sscanf(p, "%lld", out) != 1) return 0;
        if (*out < 0) *out += 4294967296LL;
        return 1;
    }

    prefix = "Maximum flow from nodes ";
    n = strlen(prefix);
    if (strncmp(line, prefix, n) != 0) return 0;

    const char *p = strrchr(line, ':');
    if (p == NULL) return 0;
    p++;
    while (*p == '\t' || *p == ' ') p++;
    if (sscanf(p, "%lld", out) != 1) return 0;
    if (*out < 0) *out += 4294967296LL;
    return 1;
}

static int parseEclMedianRuntimeLine(const char *line, double *outMs) {
    double seconds = 0.0;
    if (sscanf(line, "median runtime: %lfs", &seconds) != 1) return 0;
    *outMs = seconds * 1000.0;
    return 1;
}

static void parseRunOutput(RunResult *result) {
    char *save = NULL;
    char *line = strtok_r(result->output, "\n", &save);
    while (line != NULL) {
        long long flow = 0;
        if (sscanf(line, "flow:%lld", &flow) == 1 ||
            sscanf(line, "flow:\t%lld", &flow) == 1 ||
            sscanf(line, "flow:\t\t%lld", &flow) == 1 ||
            parseEclMaxFlowLine(line, &flow)) {
            result->flow = flow;
        }

        parseMetricLine(line, "time read", &result->timeReadMs);
        parseMetricLine(line, "time init", &result->timeInitMs);
        parseMetricLine(line, "time solve", &result->timeSolveMs);
        parseIntMetricLine(line, "block size", &result->blockSize);
        parseEclMedianRuntimeLine(line, &result->timeSolveMs);
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
        if (pollRc > 0 && (pfd.revents & POLLIN) && used < sizeof(result->output) - 1) {
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

    while (used < sizeof(result->output) - 1) {
        ssize_t got = read(pipeFd[0],
                           result->output + used,
                           sizeof(result->output) - used - 1);
        if (got <= 0) break;
        used += (size_t)got;
    }

    close(pipeFd[0]);
    result->output[used] = '\0';

    if (!result->timedOut && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        result->ok = 1;
        parseRunOutput(result);
    }

    return result->ok ? 0 : -1;
}

static int runTidalSeqPruningChild(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "USAGE: %s --run-tidal-pruning <graph.egr> <source> <sink> <mode>\n", argv[0]);
        return 1;
    }

    const char *graphPath = argv[2];
    int source = atoi(argv[3]);
    int sink = atoi(argv[4]);
    const char *mode = argv[5];

    Graph graph;
    long long readStart = nowMillis();
    if (buildGraphFromECLFileWithMode(&graph, graphPath, source, reverseModeFor(mode)) != 0) {
        return 1;
    }
    long long readEnd = nowMillis();

    long long solveStart = nowMillis();
    TidalFlowSeqStats stats;
    long long flow = tidalFlowSeq_PRUNING_STATS(&graph, source, sink, &stats);
    long long solveEnd = nowMillis();
    freeGraph(&graph);

    if (flow < 0) return 1;

    printf("solver:\t\ttidalFlowSeq_PRUNING\n");
    printf("filename:\t%s\n", graphPath);
    printf("flow:\t\t%lld\n", flow);
    printf("time read:\t%lld ms\n", readEnd - readStart);
    printf("time init:\t0 ms\n");
    printf("time solve:\t%lld ms\n", solveEnd - solveStart);
    printf("block size:\t0\n");
    return 0;
}

static const char *blockSizeModeFor(const SolverCase *solver) {
    if (solver->supportsBlockSizes) return "tunable";
    if (solver->fixedBlockSize > 0) return "fixed";
    return "not_applicable";
}

static void printRawCsvHeader(void) {
    printf("case,path,mode,nodes,edges,min_degree,avg_degree,max_degree,source,sink,solver,solver_type,block_size,block_size_mode,run,status,max_flow,time_read_s,time_init_s,time_solve_s\n");
}

static void printRawCsvRow(const BenchmarkCase *benchmarkCase, const CaseMeta *meta,
                           const SolverCase *solver, int blockSize, int run,
                           const char *status, const RunResult *result) {
    printf("%s,%s,%s,%d,%d,%d,%.6f,%d,%d,%d,%s,%s,%d,%s,%d,%s,%lld,%.6f,%.6f,%.6f\n",
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
           solver->label,
           solver->type,
           blockSize,
           blockSizeModeFor(solver),
           run,
           status,
           result->flow,
           result->timeReadMs / 1000.0,
           result->timeInitMs / 1000.0,
           result->timeSolveMs / 1000.0);
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
    printf("case,path,mode,nodes,edges,min_degree,avg_degree,max_degree,source,sink,solver,solver_type,block_size,block_size_mode,ok_runs,total_runs,max_flow,flow_consistent,median_s,avg_s,min_s,max_s,status\n");
}

static void printSummaryCsvRow(const BenchmarkCase *benchmarkCase, const CaseMeta *meta,
                               const SolverCase *solver, int blockSize, int okRuns, int totalRuns,
                               long long flow, int flowConsistent, double median, double avg,
                               double min, double max, const char *status) {
    printf("%s,%s,%s,%d,%d,%d,%.6f,%d,%d,%d,%s,%s,%d,%s,%d,%d,%lld,%s,%.6f,%.6f,%.6f,%.6f,%s\n",
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
           solver->label,
           solver->type,
           blockSize,
           blockSizeModeFor(solver),
           okRuns,
           totalRuns,
           flow,
           flowConsistent ? "yes" : "no",
           median / 1000.0,
           avg / 1000.0,
           min / 1000.0,
           max / 1000.0,
           status);
    fflush(stdout);
}

static void printSummaryHeader(void) {
    printf("%-22s %-24s %-8s %-15s %-22s %7s %14s %-15s %13s %13s %13s %13s %s\n",
           "case",
           "mode",
           "block",
           "type",
           "solver",
           "ok/runs",
           "max_flow",
           "flow_consistent",
           "median_s",
           "avg_s",
           "min_s",
           "max_s",
           "status");
}

static void printSummaryRow(const BenchmarkCase *benchmarkCase, const SolverCase *solver, int blockSize,
                            int okRuns, int totalRuns, long long flow, int flowConsistent,
                            double median, double avg, double min, double max, const char *status) {
    char runText[32];
    snprintf(runText, sizeof(runText), "%d/%d", okRuns, totalRuns);
    printf("%-22s %-24s %-8d %-15s %-22s %7s %14lld %-15s %13.6f %13.6f %13.6f %13.6f %s\n",
           benchmarkCase->name,
           benchmarkCase->mode,
           blockSize,
           solver->type,
           solver->label,
           runText,
           flow,
           flowConsistent ? "yes" : "no",
           median / 1000.0,
           avg / 1000.0,
           min / 1000.0,
           max / 1000.0,
           status);
    fflush(stdout);
}

static void printRunInfo(int runs, int timeoutSec, const int *blockSizes, int blockSizeCount,
                         int summaryCsv, int rawCsv, int caseLimit) {
    char host[256] = "unknown";
    if (gethostname(host, sizeof(host)) != 0) snprintf(host, sizeof(host), "unknown");
    host[sizeof(host) - 1] = '\0';

    time_t now = time(NULL);
    char timeText[64] = "unknown";
    struct tm tmValue;
    if (now != (time_t)-1 && localtime_r(&now, &tmValue) != NULL) {
        strftime(timeText, sizeof(timeText), "%Y-%m-%d %H:%M:%S %Z", &tmValue);
    }

    fprintf(stderr, "CUDA benchmark configuration\n");
    fprintf(stderr, "- run timestamp: %s\n", timeText);
    fprintf(stderr, "- hostname: %s\n", host);
    fprintf(stderr, "- graph cases: %zu\n", sizeof(benchmarkCases) / sizeof(benchmarkCases[0]));
    if (caseLimit > 0) fprintf(stderr, "- local case limit: %d (%s)\n", caseLimit, CUDA_BENCH_CASE_LIMIT_ENV);
    fprintf(stderr, "- solvers: %zu\n", sizeof(solvers) / sizeof(solvers[0]));
    fprintf(stderr, "- runs per solver/graph/block: %d (%s)\n", runs, CUDA_BENCH_RUNS_ENV);
    fprintf(stderr, "- default timeout per run: %d seconds (%s)\n", timeoutSec, CUDA_BENCH_TIMEOUT_ENV);
    fprintf(stderr, "- block sizes: ");
    for (int i = 0; i < blockSizeCount; i++) fprintf(stderr, "%s%d", i == 0 ? "" : ",", blockSizes[i]);
    fprintf(stderr, "\n");
    fprintf(stderr, "- output mode: %s\n", rawCsv ? "raw csv" : (summaryCsv ? "summary csv" : "summary table"));
    fprintf(stderr, "- timing: seconds, solve time only; read/init excluded\n");
    fprintf(stderr, "- benchmark build: %s %s\n", __DATE__, __TIME__);
#ifdef __VERSION__
    fprintf(stderr, "- C compiler: %s\n", __VERSION__);
#endif
    fprintf(stderr, "- default binaries: %s, %s, %s\n",
            DEFAULT_TIDAL_BIN,
            DEFAULT_TIDAL_PREFIX_BIN,
            DEFAULT_ECL_MAXFLOW_BIN);
    fprintf(stderr, "- default dinitz binary: %s\n", DEFAULT_DINITZ_BIN);
}

static int runBenchmark(const char *selfPath) {
    int failures = 0;
    int runs = envPositiveInt(CUDA_BENCH_RUNS_ENV, DEFAULT_RUNS);
    int timeoutSec = envPositiveInt(CUDA_BENCH_TIMEOUT_ENV, DEFAULT_TIMEOUT_SEC);
    int summaryCsv = envFlag(CUDA_BENCH_CSV_ENV);
    int rawCsv = envFlag(CUDA_BENCH_RAW_CSV_ENV);
    int caseLimit = envInt(CUDA_BENCH_CASE_LIMIT_ENV, 0);

    int blockSizes[MAX_BLOCK_OPTIONS];
    int blockSizeCount = 0;
    if (parseBlockSizeList(blockSizes, &blockSizeCount) != 0) return 1;

    printRunInfo(runs, timeoutSec, blockSizes, blockSizeCount, summaryCsv, rawCsv, caseLimit);

    if (rawCsv) printRawCsvHeader();
    else if (summaryCsv) printSummaryCsvHeader();
    else printSummaryHeader();

    size_t totalCases = sizeof(benchmarkCases) / sizeof(benchmarkCases[0]);
    if (caseLimit > 0 && (size_t)caseLimit < totalCases) totalCases = (size_t)caseLimit;

    for (size_t c = 0; c < totalCases; c++) {
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
            int solverBlockCount = solvers[s].supportsBlockSizes ? blockSizeCount : 1;
            for (int bi = 0; bi < solverBlockCount; bi++) {
                int blockSize = solvers[s].supportsBlockSizes ? blockSizes[bi] : solvers[s].fixedBlockSize;
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
                const char *solverStatus = "ok";

                for (int run = 1; run <= runs; run++) {
                    char sourceText[32];
                    char sinkText[32];
                    char blockText[32];
                    char eclRunsText[32];
                    snprintf(sourceText, sizeof(sourceText), "%d", meta.source);
                    snprintf(sinkText, sizeof(sinkText), "%d", meta.sink);
                    snprintf(blockText, sizeof(blockText), "%d", blockSize);
                    snprintf(eclRunsText, sizeof(eclRunsText), "%d", 1);

                    const char *tidalBin = envString(CUDA_TIDAL_BIN_ENV, DEFAULT_TIDAL_BIN);
                    char *tidalArgv[] = {
                        (char *)tidalBin,
                        (char *)benchmarkCases[c].egrPath,
                        sourceText,
                        sinkText,
                        (char *)benchmarkCases[c].mode,
                        blockText,
                        NULL,
                    };
                    const char *tidalPrefixBin = envString(CUDA_TIDAL_PREFIX_BIN_ENV, DEFAULT_TIDAL_PREFIX_BIN);
                    char *tidalPrefixArgv[] = {
                        (char *)tidalPrefixBin,
                        (char *)benchmarkCases[c].egrPath,
                        sourceText,
                        sinkText,
                        (char *)benchmarkCases[c].mode,
                        blockText,
                        NULL,
                    };
                    const char *eclBin = envString(CUDA_ECL_MAXFLOW_BIN_ENV, DEFAULT_ECL_MAXFLOW_BIN);
                    char *eclArgv[] = {
                        (char *)eclBin,
                        (char *)benchmarkCases[c].egrPath,
                        sourceText,
                        sinkText,
                        eclRunsText,
                        NULL,
                    };
                    const char *dinitzBin = envString(CUDA_DINITZ_BIN_ENV, DEFAULT_DINITZ_BIN);
                    char *dinitzArgv[] = {
                        (char *)dinitzBin,
                        (char *)benchmarkCases[c].egrPath,
                        sourceText,
                        sinkText,
                        (char *)benchmarkCases[c].mode,
                        NULL,
                    };
                    char *tidalSeqArgv[] = {
                        (char *)selfPath,
                        (char *)"--run-tidal-pruning",
                        (char *)benchmarkCases[c].egrPath,
                        sourceText,
                        sinkText,
                        (char *)benchmarkCases[c].mode,
                        NULL,
                    };

                    char *const *argv = NULL;
                    switch (solvers[s].kind) {
                        case SOLVER_TIDAL_CUDA:
                            argv = tidalArgv;
                            break;
                        case SOLVER_TIDAL_CUDA_PREFIX:
                            argv = tidalPrefixArgv;
                            break;
                        case SOLVER_ECL_MAXFLOW:
                            argv = eclArgv;
                            break;
                        case SOLVER_DINITZ:
                            argv = dinitzArgv;
                            break;
                        case SOLVER_TIDAL_SEQ_PRUNING:
                            argv = tidalSeqArgv;
                            break;
                    }

                    RunResult result;
                    int ok = (runProgram((char *const *)argv, timeoutSec, &result) == 0);
                    if (result.blockSize == 0) result.blockSize = blockSize;

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
                    }

                    if (rawCsv) {
                        printRawCsvRow(&benchmarkCases[c], &meta, &solvers[s], blockSize, run, status, &result);
                    }

                    if (!ok && result.output[0] != '\0') {
                        fprintf(stderr, "Output from failed run (%s / %s / block=%d):\n%s\n",
                                benchmarkCases[c].name,
                                solvers[s].label,
                                blockSize,
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
                    if (solverFailed == 0 && okRuns != runs) solverStatus = "partial";
                    int flowConsistent = (okRuns > 0 && expectedFlow >= 0 && solverFlow == expectedFlow);

                    if (summaryCsv) {
                        printSummaryCsvRow(&benchmarkCases[c],
                                           &meta,
                                           &solvers[s],
                                           blockSize,
                                           okRuns,
                                           runs,
                                           solverFlow,
                                           flowConsistent,
                                           median,
                                           avg,
                                           minSolve,
                                           maxSolve,
                                           solverStatus);
                    } else {
                        printSummaryRow(&benchmarkCases[c],
                                        &solvers[s],
                                        blockSize,
                                        okRuns,
                                        runs,
                                        solverFlow,
                                        flowConsistent,
                                        median,
                                        avg,
                                        minSolve,
                                        maxSolve,
                                        solverStatus);
                    }
                }

                free(solveTimes);
            }
        }
    }

    if (failures != 0) {
        fflush(stdout);
        printf("CUDA benchmark finished with %d failed run(s).\n", failures);
        return 1;
    }

    fflush(stdout);
    printf("CUDA benchmark finished successfully.\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--run-tidal-pruning") == 0) {
        return runTidalSeqPruningChild(argc, argv);
    }

    return runBenchmark(argv[0]);
}
