#ifndef SEQUENTIAL_BENCHMARK_COMMON_H
#define SEQUENTIAL_BENCHMARK_COMMON_H

#include <chrono>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sequential_benchmark {

using Clock = std::chrono::high_resolution_clock;
using Node = uint32_t;
using Capacity = uint64_t;

struct Arc {
    Node from = 0;
    Node to = 0;
    Capacity capacity = 0;
};

struct ParsedGraph {
    Node nodeCount = 0;
    Node source = 0;
    Node sink = 0;
    std::vector<Arc> arcs;
};

struct TimedParsedGraph {
    ParsedGraph graph;
    long long timeReadMs = 0;
};

inline long long millis(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

inline Node parseNodeArg(const char *text, const char *name) {
    char *end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 || value > UINT32_MAX) {
        throw std::runtime_error(std::string("Invalid ") + name + ": " + text);
    }
    return static_cast<Node>(value);
}

inline void validateNode(Node node, Node nodeCount, const std::string &name) {
    if (node >= nodeCount) {
        throw std::runtime_error(name + " out of range");
    }
}

inline void validateMode(const std::string &mode) {
    if (mode != "auto" && mode != "addReverseEdges" && mode != "useExistingReverseEdges") {
        throw std::runtime_error("Invalid mode '" + mode + "'. Use: auto | addReverseEdges | useExistingReverseEdges");
    }
}

template <typename T>
void readExact(std::ifstream &in, T *dst, size_t count, const std::string &what) {
    if (count == 0) return;
    in.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(count * sizeof(T)));
    if (in.gcount() != static_cast<std::streamsize>(count * sizeof(T))) {
        throw std::runtime_error("Failed to read EGR " + what);
    }
}

inline ParsedGraph readDimacsMaxFlow(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Unable to open DIMACS file: " + path);
    }

    ParsedGraph result;
    char type = '\0';
    std::string token;
    int edgeCount = 0;
    int parsedEdges = 0;
    int source = -1;
    int sink = -1;

    while (in >> type) {
        if (type == 'c') {
            std::getline(in, token);
        } else if (type == 'p') {
            in >> token;
            if (token != "max") {
                throw std::runtime_error("Expected DIMACS 'p max' problem line");
            }
            int nodes = 0;
            in >> nodes >> edgeCount;
            if (nodes <= 0 || edgeCount < 0) {
                throw std::runtime_error("Invalid DIMACS problem size");
            }
            result.nodeCount = static_cast<Node>(nodes);
            result.arcs.reserve(static_cast<size_t>(edgeCount));
        } else if (type == 'n') {
            int node = 0;
            char role = '\0';
            in >> node >> role;
            if (node <= 0) {
                throw std::runtime_error("Invalid DIMACS terminal node");
            }
            if (role == 's') source = node - 1;
            else if (role == 't') sink = node - 1;
        } else if (type == 'a') {
            int from = 0;
            int to = 0;
            long long capacity = 0;
            in >> from >> to >> capacity;
            if (result.nodeCount == 0) {
                throw std::runtime_error("DIMACS arc appeared before problem line");
            }
            if (from <= 0 || to <= 0 ||
                static_cast<Node>(from) > result.nodeCount ||
                static_cast<Node>(to) > result.nodeCount ||
                capacity < 0) {
                throw std::runtime_error("Invalid DIMACS arc row");
            }
            result.arcs.push_back({static_cast<Node>(from - 1),
                                   static_cast<Node>(to - 1),
                                   static_cast<Capacity>(capacity)});
            parsedEdges++;
        } else {
            std::getline(in, token);
        }
    }

    if (source < 0 || sink < 0) {
        throw std::runtime_error("DIMACS file is missing source/sink node lines");
    }
    if (parsedEdges != edgeCount) {
        throw std::runtime_error("DIMACS edge count does not match parsed edge rows");
    }

    result.source = static_cast<Node>(source);
    result.sink = static_cast<Node>(sink);
    validateNode(result.source, result.nodeCount, "source");
    validateNode(result.sink, result.nodeCount, "sink");
    return result;
}

inline ParsedGraph readEgrMaxFlow(const std::string &path, Node source, int sinkArg,
                                  const std::string &mode) {
    validateMode(mode);

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Unable to open EGR file: " + path);
    }

    int nodes = 0;
    int edges = 0;
    readExact(in, &nodes, 1, "node count");
    readExact(in, &edges, 1, "edge count");
    if (nodes < 1 || edges < 0) {
        throw std::runtime_error("Invalid EGR header in: " + path);
    }

    std::vector<int> nindex(static_cast<size_t>(nodes) + 1);
    std::vector<int> nlist(static_cast<size_t>(edges));
    readExact(in, nindex.data(), nindex.size(), "index array");
    readExact(in, nlist.data(), nlist.size(), "neighbor array");

    std::vector<int> weights(static_cast<size_t>(edges));
    bool hasWeights = false;
    if (edges > 0) {
        in.read(reinterpret_cast<char *>(weights.data()),
                static_cast<std::streamsize>(weights.size() * sizeof(int)));
        const std::streamsize got = in.gcount();
        if (got == static_cast<std::streamsize>(weights.size() * sizeof(int))) {
            hasWeights = true;
        } else if (got != 0) {
            throw std::runtime_error("Corrupt EGR weight section in: " + path);
        }
    }

    if (nindex.front() != 0 || nindex.back() != edges) {
        throw std::runtime_error("Invalid EGR index bounds in: " + path);
    }
    for (int i = 1; i <= nodes; i++) {
        if (nindex[i] < nindex[i - 1]) {
            throw std::runtime_error("Non-monotone EGR index array in: " + path);
        }
    }

    const Node nodeCount = static_cast<Node>(nodes);
    const Node sink = sinkArg < 0 ? static_cast<Node>(nodes - 1) : static_cast<Node>(sinkArg);
    validateNode(source, nodeCount, "source");
    validateNode(sink, nodeCount, "sink");
    if (source == sink) {
        throw std::runtime_error("source and sink must differ");
    }

    ParsedGraph result;
    result.nodeCount = nodeCount;
    result.source = source;
    result.sink = sink;
    result.arcs.reserve(static_cast<size_t>(edges));

    if (!hasWeights) std::srand(static_cast<unsigned int>(source));
    for (int u = 0; u < nodes; u++) {
        for (int e = nindex[u]; e < nindex[u + 1]; e++) {
            const int v = nlist[e];
            if (v < 0 || v >= nodes) {
                throw std::runtime_error("EGR endpoint out of range in: " + path);
            }
            long long cap = hasWeights ? weights[e] : (std::rand() % nodes);
            if (cap < 0) cap = -cap;
            result.arcs.push_back({static_cast<Node>(u),
                                   static_cast<Node>(v),
                                   static_cast<Capacity>(cap)});
        }
    }

    return result;
}

inline TimedParsedGraph readGraphWithOptionalEgrTerminals(int argc, char **argv) {
    Node source = 0;
    int sink = -1;
    std::string mode = "auto";

    auto start = Clock::now();
    ParsedGraph graph;
    if (argc == 2) {
        graph = readDimacsMaxFlow(argv[1]);
    } else {
        if (argc == 3) {
            mode = argv[2];
        } else {
            source = parseNodeArg(argv[2], "source");
            sink = static_cast<int>(parseNodeArg(argv[3], "sink"));
            if (argc == 5) mode = argv[4];
        }
        graph = readEgrMaxFlow(argv[1], source, sink, mode);
    }
    auto end = Clock::now();
    return {std::move(graph), millis(start, end)};
}

inline TimedParsedGraph readGraphWithRequiredEgrTerminals(int argc, char **argv) {
    auto start = Clock::now();
    ParsedGraph graph;
    if (argc == 2) {
        graph = readDimacsMaxFlow(argv[1]);
    } else {
        std::string mode = argc == 5 ? argv[4] : "auto";
        graph = readEgrMaxFlow(argv[1],
                               parseNodeArg(argv[2], "source"),
                               static_cast<int>(parseNodeArg(argv[3], "sink")),
                               mode);
    }
    auto end = Clock::now();
    return {std::move(graph), millis(start, end)};
}

template <typename Flow>
void printResult(const std::string &solverName, const std::string &filename, Flow flow,
                 long long timeReadMs, long long timeInitMs, long long timeSolveMs) {
    std::cout << "solver:\t\t" << solverName << '\n'
              << "filename:\t" << filename << '\n'
              << "flow:\t\t" << flow << '\n'
              << "time read:\t" << timeReadMs << " ms\n"
              << "time init:\t" << timeInitMs << " ms\n"
              << "time solve:\t" << timeSolveMs << " ms\n"
              << "# of threads:\t1\n";
}

} // namespace sequential_benchmark

#endif // SEQUENTIAL_BENCHMARK_COMMON_H
