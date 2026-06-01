#ifndef BENCHMARK_CASES_H
#define BENCHMARK_CASES_H

#define GRAPH(name) "benchmarks/graphs/" name ".egr"

static const BenchmarkCase benchmarkCases[] = {
    {"ac_n4000", GRAPH("ac_n4000"), "addReverseEdges", 0, -1},
    {"ac_n8000", GRAPH("ac_n8000"), "addReverseEdges", 0, -1},

    {"amazon0312A", GRAPH("amazon0312"), "useExistingReverseEdges", 32, 335},
    {"amazon0312B", GRAPH("amazon0312"), "useExistingReverseEdges", 44038, 9363},

    {"in-2004A", GRAPH("in-2004"), "useExistingReverseEdges", 854554, 246034},
    {"in-2004B", GRAPH("in-2004"), "useExistingReverseEdges", 687758, 863247},

    {"as-skitterA", GRAPH("as-skitter"), "useExistingReverseEdges", 7046, 1039},
    {"as-skitterB", GRAPH("as-skitter"), "useExistingReverseEdges", 811, 7581},

    {"soc-LiveJournal1A", GRAPH("soc-LiveJournal1"), "useExistingReverseEdges", 10009, 37344},
    {"soc-LiveJournal1B", GRAPH("soc-LiveJournal1"), "useExistingReverseEdges", 39283, 91282},

    {"cit-PatentsA", GRAPH("cit-Patents"), "useExistingReverseEdges", 2506522, 3559277},
    {"cit-PatentsB", GRAPH("cit-Patents"), "useExistingReverseEdges", 2131075, 2342725},

    {"USA-road-d.NYA", GRAPH("USA-road-d.NY"), "useExistingReverseEdges", 140960, 134677},
    {"USA-road-d.NYB", GRAPH("USA-road-d.NY"), "useExistingReverseEdges", 194677, 47619},

    {"2d-2e20.symA", GRAPH("2d-2e20.sym"), "useExistingReverseEdges", 0, 400000},
    {"2d-2e20.symB", GRAPH("2d-2e20.sym"), "useExistingReverseEdges", 400000, 800000},

    {"rmat22.symA", GRAPH("rmat22.sym"), "useExistingReverseEdges", 1837250, 2244496},
    {"rmat22.symB", GRAPH("rmat22.sym"), "useExistingReverseEdges", 3414666, 1395919},

    {"kron_g500-logn21A", GRAPH("kron_g500-logn21"), "useExistingReverseEdges", 1421105, 1531673},
    {"kron_g500-logn21B", GRAPH("kron_g500-logn21"), "useExistingReverseEdges", 773179, 1058878},

    // EXTREMELY SLOW
    // {"washington_RLG_L_r896_c1792", GRAPH("washington_RLG_L_r896_c1792"), "addReverseEdges", 0, -1},
    // {"washington_RLG_L_r1280_c2560", GRAPH("washington_RLG_L_r1280_c2560"), "addReverseEdges", 0, -1},
};

#undef GRAPH

#endif
