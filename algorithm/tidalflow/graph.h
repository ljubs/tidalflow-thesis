#ifndef GRAPH_H
#define GRAPH_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int u, v, c;
} Edge;

typedef struct {
    int nodeCount;
    int edgeCount;

    int *ver;
    int *src;
    int *dst;
    int *rev;
    int *capacity;
    int *flow;
} Graph;

#define ECL_REVERSE_REQUIRE_EXISTING 0
#define ECL_REVERSE_ADD_RESIDUAL 1
#define ECL_REVERSE_AUTO 2

void initGraph(Graph *g, int n, int m);
void buildGraphFromStdin(Graph *g);
int buildGraphFromECLFile(Graph *g, const char *path, int source);
int buildGraphFromECLFileWithMode(Graph *g, const char *path, int source, int reverseMode);
void buildGraph(Graph *g);
void freeGraph(Graph *g);

#ifdef __cplusplus
}
#endif

#endif // GRAPH_H
