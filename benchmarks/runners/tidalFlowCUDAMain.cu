#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graph.h"

#ifndef TIDAL_FLOW_CUDA_SOLVER_NAME
#define TIDAL_FLOW_CUDA_SOLVER_NAME "tidalFlowCUDA"
#endif

#ifndef TIDAL_FLOW_CUDA_RUNNER
#define TIDAL_FLOW_CUDA_RUNNER runTidalFlowCUDAInstanceWithBlockSize
#endif

extern "C" int TIDAL_FLOW_CUDA_RUNNER(const char *graphPath,
                                      int source,
                                      int sink,
                                      int mode,
                                      int blockSize,
                                      double *totalTimeOut,
                                      long long *maxFlowOut);

static int parseReverseModeName(const char *name, int *modeOut) {
    if (strcmp(name, "auto") == 0) {
        *modeOut = ECL_REVERSE_AUTO;
        return 0;
    }
    if (strcmp(name, "addReverseEdges") == 0) {
        *modeOut = ECL_REVERSE_ADD_RESIDUAL;
        return 0;
    }
    if (strcmp(name, "useExistingReverseEdges") == 0) {
        *modeOut = ECL_REVERSE_REQUIRE_EXISTING;
        return 0;
    }
    return -1;
}

static int parseBlockSizeName(const char *name, int *blockSizeOut) {
    int blockSize = atoi(name);
    if (blockSize == 128 || blockSize == 256 || blockSize == 512 || blockSize == 1024) {
        *blockSizeOut = blockSize;
        return 0;
    }
    return -1;
}

int main(int argc, char *argv[]) {
    if (argc < 4 || argc > 6) {
        fprintf(stderr,
                "USAGE: %s <graph.egr> <source> <sink> [mode] [block_size]\n"
                "  mode: auto (default), addReverseEdges, useExistingReverseEdges\n"
                "  block_size: 128, 256, 512, 1024 (default 256)\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *graphPath = argv[1];
    const int source = atoi(argv[2]);
    const int sink = atoi(argv[3]);
    int mode = ECL_REVERSE_AUTO;
    int blockSize = 256;

    if (argc == 5) {
        if (parseReverseModeName(argv[4], &mode) != 0 &&
            parseBlockSizeName(argv[4], &blockSize) != 0) {
            fprintf(stderr,
                    "Invalid mode/block_size '%s'. Use mode auto|addReverseEdges|useExistingReverseEdges or block size 128|256|512|1024\n",
                    argv[4]);
            return EXIT_FAILURE;
        }
    }
    if (argc == 6) {
        if (parseReverseModeName(argv[4], &mode) != 0) {
            fprintf(stderr, "Invalid mode '%s'. Use: auto | addReverseEdges | useExistingReverseEdges\n", argv[4]);
            return EXIT_FAILURE;
        }
        if (parseBlockSizeName(argv[5], &blockSize) != 0) {
            fprintf(stderr, "Invalid block_size '%s'. Use: 128 | 256 | 512 | 1024\n", argv[5]);
            return EXIT_FAILURE;
        }
    }

    double totalTime = 0.0;
    long long maxFlow = 0;
    if (TIDAL_FLOW_CUDA_RUNNER(graphPath, source, sink, mode, blockSize, &totalTime, &maxFlow) != 0) {
        fprintf(stderr, "Failed to run %s on %s\n", TIDAL_FLOW_CUDA_SOLVER_NAME, graphPath);
        return EXIT_FAILURE;
    }

    printf("solver:\t\t%s\n", TIDAL_FLOW_CUDA_SOLVER_NAME);
    printf("filename:\t%s\n", graphPath);
    printf("flow:\t\t%lld\n", maxFlow);
    printf("time read:\t0 ms\n");
    printf("time init:\t0 ms\n");
    printf("time solve:\t%.3f ms\n", totalTime * 1000.0);
    printf("block size:\t%d\n", blockSize);
    return 0;
}
