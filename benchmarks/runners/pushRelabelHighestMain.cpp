#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "benchmark_common.h"
#include "algorithms/sequential/push_relabel_highest.h"

namespace {

using sequential_benchmark::Capacity;
using sequential_benchmark::Clock;
using sequential_benchmark::Node;
using PushRelabelGraph = std::vector<std::vector<cached_edge<Node, Capacity>>>;

void setReverseEdgeCap(PushRelabelGraph &graph) {
    for (auto &edges : graph) {
        for (auto &edge : edges) {
            edge.reverse_r_capacity = graph[edge.dst_vertex][edge.reverse_edge_index].r_capacity;
        }
    }
}

uint64_t edgeKey(Node from, Node to) {
    return (static_cast<uint64_t>(from) << 32) | static_cast<uint64_t>(to);
}

struct ResidualEdgePair {
    Node from = 0;
    Node to = 0;
    Capacity forwardCapacity = 0;
    Capacity reverseCapacity = 0;
};

class ResidualEdgeBuilder {
public:
    ResidualEdgeBuilder(Node nodes, size_t expectedArcs) : nodeCount(nodes) {
        pairs.reserve(expectedArcs);
        pairIndex.reserve(expectedArcs);
    }

    void addArc(Node from, Node to, Capacity capacity) {
        const uint64_t forwardKey = edgeKey(from, to);
        auto forward = pairIndex.find(forwardKey);
        if (forward != pairIndex.end()) {
            pairs[forward->second].forwardCapacity += capacity;
            return;
        }

        auto reverse = pairIndex.find(edgeKey(to, from));
        if (reverse != pairIndex.end()) {
            pairs[reverse->second].reverseCapacity += capacity;
            return;
        }

        pairIndex.emplace(forwardKey, pairs.size());
        pairs.push_back({from, to, capacity, 0});
    }

    PushRelabelGraph build() const {
        PushRelabelGraph graph(static_cast<size_t>(nodeCount));
        std::vector<size_t> degree(static_cast<size_t>(nodeCount), 0);
        for (const auto &pair : pairs) {
            degree[pair.from]++;
            degree[pair.to]++;
        }
        for (size_t i = 0; i < graph.size(); i++) {
            graph[i].reserve(degree[i]);
        }

        for (const auto &pair : pairs) {
            const Node forwardIndex = static_cast<Node>(graph[pair.from].size());
            const Node reverseIndex = static_cast<Node>(graph[pair.to].size());
            graph[pair.from].push_back(cached_edge<Node, Capacity>(pair.to,
                                                                    pair.forwardCapacity,
                                                                    reverseIndex));
            graph[pair.to].push_back(cached_edge<Node, Capacity>(pair.from,
                                                                  pair.reverseCapacity,
                                                                  forwardIndex));
        }
        setReverseEdgeCap(graph);
        return graph;
    }

private:
    Node nodeCount;
    std::vector<ResidualEdgePair> pairs;
    std::unordered_map<uint64_t, size_t> pairIndex;
};

PushRelabelGraph buildGraph(const sequential_benchmark::ParsedGraph &input) {
    ResidualEdgeBuilder builder(input.nodeCount, input.arcs.size());
    for (const auto &arc : input.arcs) {
        builder.addArc(arc.from, arc.to, arc.capacity);
    }
    return builder.build();
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
        const std::string solverName = argc == 2 ? "push-relabel-highest-dimacs"
                                                 : "push-relabel-highest-egr";

        auto initStart = Clock::now();
        push_relabel_highest::max_flow_instance<std::vector, Node, Capacity> solver(std::move(graph),
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
