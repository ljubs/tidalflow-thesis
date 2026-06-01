#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "benchmark_common.h"
#include "algorithms/sequential/dinic.h"

namespace {

using sequential_benchmark::Capacity;
using sequential_benchmark::Clock;
using sequential_benchmark::Node;
using DinicGraph = std::vector<std::vector<basic_edge<Node, Capacity>>>;

void addDirectedEdge(DinicGraph &graph, Node from, Node to, Capacity capacity) {
    const Node forwardIndex = static_cast<Node>(graph[from].size());
    const Node reverseIndex = static_cast<Node>(graph[to].size());
    graph[from].push_back(basic_edge<Node, Capacity>(to, capacity, reverseIndex));
    graph[to].push_back(basic_edge<Node, Capacity>(from, 0, forwardIndex));
}

DinicGraph buildGraph(const sequential_benchmark::ParsedGraph &input) {
    DinicGraph graph(static_cast<size_t>(input.nodeCount));
    for (const auto &arc : input.arcs) {
        addDirectedEdge(graph, arc.from, arc.to, arc.capacity);
    }
    return graph;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3 && argc != 4 && argc != 5) {
        std::cerr << "USAGE: " << argv[0] << " <graph.max>\n"
                  << "       " << argv[0] << " <graph.egr> [mode]\n"
                  << "       " << argv[0] << " <graph.egr> <source> <sink> [mode]\n"
                  << "  mode: auto (default), addReverseEdges, useExistingReverseEdges\n";
        return 1;
    }

    try {
        auto input = sequential_benchmark::readGraphWithOptionalEgrTerminals(argc, argv);
        auto graph = buildGraph(input.graph);
        const std::string solverName = argc == 2 ? "dinic-dimacs" : "dinic-egr";

        auto initStart = Clock::now();
        dinic::max_flow_instance<std::vector, Node, Capacity> solver(std::move(graph),
                                                                     input.graph.source,
                                                                     input.graph.sink);
        auto initEnd = Clock::now();

        auto solveStart = Clock::now();
        Capacity flow = solver.find_max_flow();
        auto solveEnd = Clock::now();

        sequential_benchmark::printResult(solverName,
                                          argv[1],
                                          flow,
                                          input.timeReadMs,
                                          sequential_benchmark::millis(initStart, initEnd),
                                          sequential_benchmark::millis(solveStart, solveEnd));
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    return 0;
}
