#include <climits>
#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "benchmark_common.h"
#include "Dinics.h"

namespace {

using sequential_benchmark::Clock;

EdgeList buildGraph(const sequential_benchmark::ParsedGraph &input) {
    if (input.nodeCount > INT_MAX) {
        throw std::runtime_error("Dinitz driver only supports INT_MAX vertices");
    }

    EdgeList graph;
    graph.n = static_cast<int>(input.nodeCount);
    graph.isDirected = true;
    graph.hasWeights = true;
    graph.edges.reserve(input.arcs.size());
    graph.weights.reserve(input.arcs.size());

    for (const auto &arc : input.arcs) {
        graph.edges.emplace_back(static_cast<int>(arc.from), static_cast<int>(arc.to));
        graph.weights.push_back(static_cast<double>(arc.capacity));
    }

    return graph;
}

long long roundedFlow(double flow) {
    return static_cast<long long>(std::llround(flow));
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2 && argc != 4 && argc != 5) {
        std::cerr << "USAGE: " << argv[0] << " <graph.max>\n"
                  << "       " << argv[0] << " <graph.egr> <source> <sink> [mode]\n"
                  << "  mode: auto (default), addReverseEdges, useExistingReverseEdges\n";
        return 1;
    }

    try {
        auto input = sequential_benchmark::readGraphWithRequiredEgrTerminals(argc, argv);
        auto graph = buildGraph(input.graph);
        const std::string solverName = argc == 2 ? "dinitz4skip-dimacs" : "dinitz4skip-egr";

        auto initStart = Clock::now();
        Dinics::Dinics4Skip solver(graph);
        auto initEnd = Clock::now();

        auto solveStart = Clock::now();
        double flow = solver.maxFlow(static_cast<int>(input.graph.source),
                                     static_cast<int>(input.graph.sink));
        auto solveEnd = Clock::now();

        sequential_benchmark::printResult(solverName,
                                          argv[1],
                                          roundedFlow(flow),
                                          input.timeReadMs,
                                          sequential_benchmark::millis(initStart, initEnd),
                                          sequential_benchmark::millis(solveStart, solveEnd));
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    return 0;
}
