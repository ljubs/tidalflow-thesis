CC ?= gcc
CXX ?= g++
NVCC ?= nvcc

CFLAGS ?= -O3 -Wall -Wextra -std=c11 -Ialgorithm/tidalflow -Ibenchmarks/runners/common
OPENMP_CFLAGS ?= $(CFLAGS) -fopenmp
CXXFLAGS ?= -O3 -Wall -Wextra -std=c++17 \
	-Ibenchmarks/runners/common \
	-Ialgorithm/comparison/zagrosss-maxflow/src/lib \
	-Ialgorithm/comparison/scale-free-flow/code/include
OPENMP_CXXFLAGS ?= $(CXXFLAGS) -fopenmp
NVCCFLAGS ?= -O3 -std=c++17 -Ialgorithm/tidalflow -Xcompiler -fopenmp -arch=sm_80

SEQUENTIAL_SRCS = \
	benchmarks/runners/sequentialBenchmark.c \
	algorithm/tidalflow/graph.c \
	algorithm/tidalflow/tidalFlowSeq_BASELINE.c \
	algorithm/tidalflow/tidalFlowSeq_BLOCKING.c \
	algorithm/tidalflow/tidalFlowSeq_PRUNING.c \
	algorithm/tidalflow/tidalFlowSeq_COMBINED.c

OPENMP_SRCS = \
	benchmarks/runners/openmpBenchmark.c \
	algorithm/tidalflow/graph.c \
	algorithm/tidalflow/tidalFlowOpenMP.c \
	algorithm/tidalflow/tidalFlowSeq_PRUNING.c

CUDA_BENCHMARK_SRCS = \
	benchmarks/runners/cudaBenchmark.c \
	algorithm/tidalflow/graph.c \
	algorithm/tidalflow/tidalFlowSeq_PRUNING.c

.PHONY: all sequential openmp cuda run-sequential run-openmp run-cuda clean

all: sequential

sequential:
	mkdir -p build
	$(CC) $(CFLAGS) $(SEQUENTIAL_SRCS) -o build/sequentialBenchmark
	$(CXX) $(CXXFLAGS) benchmarks/runners/dinicMain.cpp -o build/dinic
	$(CXX) $(CXXFLAGS) benchmarks/runners/pushRelabelHighestMain.cpp -o build/pushRelabelHighest
	$(CXX) $(CXXFLAGS) -Wno-sign-compare benchmarks/runners/dinitzBidirectionalMain.cpp \
		algorithm/comparison/scale-free-flow/code/source/Dinics.cpp \
		algorithm/comparison/scale-free-flow/code/source/DinicsStats.cpp \
		-o build/dinitzBidirectional

openmp:
	mkdir -p build
	$(CC) $(OPENMP_CFLAGS) $(OPENMP_SRCS) -o build/openmpBenchmark
	$(CXX) $(CXXFLAGS) -Wno-sign-compare benchmarks/runners/dinitzBidirectionalMain.cpp \
		algorithm/comparison/scale-free-flow/code/source/Dinics.cpp \
		algorithm/comparison/scale-free-flow/code/source/DinicsStats.cpp \
		-o build/dinitzBidirectional
	$(CXX) $(OPENMP_CXXFLAGS) benchmarks/runners/pushRelabelPprMain.cpp -o build/pushRelabelPpr

cuda:
	mkdir -p build
	$(CC) $(CFLAGS) $(CUDA_BENCHMARK_SRCS) -o build/cudaBenchmark
	$(CXX) $(CXXFLAGS) -Wno-sign-compare benchmarks/runners/dinitzBidirectionalMain.cpp \
		algorithm/comparison/scale-free-flow/code/source/Dinics.cpp \
		algorithm/comparison/scale-free-flow/code/source/DinicsStats.cpp \
		-o build/dinitzBidirectional
	$(NVCC) $(NVCCFLAGS) \
		benchmarks/runners/tidalFlowCUDAMain.cu \
		algorithm/tidalflow/tidalFlowCUDA.cu \
		algorithm/tidalflow/graph.c \
		-o build/tidalFlowCUDA
	$(NVCC) $(NVCCFLAGS) \
		-DTIDAL_FLOW_CUDA_SOLVER_NAME='"tidalFlowCUDAPrefix"' \
		-DTIDAL_FLOW_CUDA_RUNNER=runTidalFlowCUDAPrefixInstanceWithBlockSize \
		benchmarks/runners/tidalFlowCUDAMain.cu \
		algorithm/tidalflow/tidalFlowCUDAPrefix.cu \
		algorithm/tidalflow/graph.c \
		-o build/tidalFlowCUDAPrefix
	$(NVCC) $(NVCCFLAGS) -Ialgorithm/comparison/ECL-MaxFlow/lib \
		algorithm/comparison/ECL-MaxFlow/src/main.cu \
		-o build/eclMaxFlow

run-sequential: sequential
	SEQ_BENCH_RUNS=3 SEQ_BENCH_TIMEOUT_SEC=1800 ./build/sequentialBenchmark

run-openmp: openmp
	OMP_BENCH_RUNS=$${OMP_BENCH_RUNS:-3} OMP_BENCH_TIMEOUT_SEC=$${OMP_BENCH_TIMEOUT_SEC:-1800} ./build/openmpBenchmark

run-cuda: cuda
	CUDA_BENCH_RUNS=$${CUDA_BENCH_RUNS:-3} CUDA_BENCH_TIMEOUT_SEC=$${CUDA_BENCH_TIMEOUT_SEC:-1800} ./build/cudaBenchmark

clean:
	rm -rf build
