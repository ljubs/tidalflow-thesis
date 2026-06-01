#include <cstdint>
#include <cassert>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "benchmark_common.h"
#include "common_types.h"
#include "algorithms/parallel/parallel_push_relabel.h"

namespace {

using sequential_benchmark::Capacity;
using sequential_benchmark::Clock;
using sequential_benchmark::Node;
using Edge = cached_edge<Node, Capacity>;
using PprGraph = std::vector<std::vector<Edge>>;

Node parseThreadCount(const char *text) {
    const auto value = sequential_benchmark::parseNodeArg(text, "thread count");
    if (value == 0) {
        throw std::runtime_error("thread count must be positive");
    }
    return value;
}

void setReverseCapacities(PprGraph &graph) {
    for (auto &edges : graph) {
        for (auto &edge : edges) {
            edge.reverse_r_capacity = graph[edge.dst_vertex][edge.reverse_edge_index].r_capacity;
        }
    }
}

PprGraph buildGraph(const sequential_benchmark::ParsedGraph &input) {
    const Node undefined = std::numeric_limits<Node>::max();
    std::vector<Edge> edges(input.arcs.size() * 2);
    std::vector<std::unordered_map<Node, Node>> edgeMap(input.nodeCount);
    std::vector<uint32_t> outgoingEdgeCount(input.nodeCount);
    size_t pos = 0;

    for (const auto &arc : input.arcs) {
        auto forward = edgeMap[arc.from].find(arc.to);
        if (forward != edgeMap[arc.from].end()) {
            edges[forward->second].r_capacity += arc.capacity;
            continue;
        }

        auto reverse = edgeMap[arc.to].find(arc.from);
        if (reverse != edgeMap[arc.to].end()) {
            edges[reverse->second + 1].r_capacity += arc.capacity;
            continue;
        }

        edges[pos] = Edge(arc.to, arc.capacity, undefined);
        edges[pos + 1] = Edge(arc.from, 0, undefined);
        ++outgoingEdgeCount[arc.from];
        ++outgoingEdgeCount[arc.to];
        edgeMap[arc.from].emplace(arc.to, static_cast<Node>(pos));
        pos += 2;
    }

    edgeMap.clear();
    edgeMap.shrink_to_fit();

    PprGraph graph(static_cast<size_t>(input.nodeCount));
    for (size_t i = 0; i < graph.size(); ++i) {
        graph[i].reserve(outgoingEdgeCount[i]);
    }

    for (size_t i = 0; i < pos; i += 2) {
        auto &edge = edges[i];
        auto &reverseEdge = edges[i + 1];
        graph[reverseEdge.dst_vertex].emplace_back(edge.dst_vertex, edge.r_capacity, static_cast<Node>(i + 1));
    }

    std::vector<uint32_t> sizes(graph.size());
    for (size_t i = 0; i < graph.size(); ++i) {
        sizes[i] = static_cast<uint32_t>(graph[i].size());
    }

    for (size_t i = 0; i < graph.size(); ++i) {
        for (size_t k = 0; k < sizes[i]; ++k) {
            auto &edge = graph[i][k];
            auto reverseEdge = edges[edge.reverse_edge_index];
            edge.reverse_edge_index = static_cast<Node>(graph[edge.dst_vertex].size());
            graph[edge.dst_vertex].emplace_back(static_cast<Node>(i), reverseEdge.r_capacity, static_cast<Node>(k));
        }
    }

    setReverseCapacities(graph);
    return graph;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) {
        std::cerr << "USAGE: " << argv[0] << " <graph.egr> <source> <sink> <threads> [mode]\n"
                  << "  mode: auto (default), addReverseEdges, useExistingReverseEdges\n";
        return 1;
    }

    try {
        const Node threadCount = parseThreadCount(argv[4]);
        char *graphArgs[5] = {argv[0], argv[1], argv[2], argv[3], argc == 6 ? argv[5] : nullptr};
        const int graphArgc = argc == 6 ? 5 : 4;

        auto input = sequential_benchmark::readGraphWithRequiredEgrTerminals(graphArgc, graphArgs);
        auto graph = buildGraph(input.graph);

        auto initStart = Clock::now();
        parallel_push_relabel::max_flow_instance<std::vector, Node, Capacity> solver(
            std::move(graph),
            input.graph.source,
            input.graph.sink,
            static_cast<size_t>(threadCount));
        auto initEnd = Clock::now();

        auto solveStart = Clock::now();
        Capacity flow = solver.find_max_flow();
        auto solveEnd = Clock::now();

        std::cout << "solver:\t\tpush-relabel-ppr\n"
                  << "filename:\t" << argv[1] << '\n'
                  << "flow:\t\t" << flow << '\n'
                  << "time read:\t" << input.timeReadMs << " ms\n"
                  << "time init:\t" << sequential_benchmark::millis(initStart, initEnd) << " ms\n"
                  << "time solve:\t" << sequential_benchmark::millis(solveStart, solveEnd) << " ms\n"
                  << "# of threads:\t" << threadCount << '\n';
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    return 0;
}
