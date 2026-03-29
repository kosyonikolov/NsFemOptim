#ifndef LIBS_GRAPHS_INC_GRAPHS_COLOR
#define LIBS_GRAPHS_INC_GRAPHS_COLOR

#include <vector>

namespace graphs
{
    bool verifyColoring(const std::vector<std::vector<int>> & graph, const std::vector<std::vector<int>> & coloring);

    std::vector<int> buildSmallestLastOrdering(const std::vector<std::vector<int>> & graph);

    std::vector<std::vector<int>> partitionGraphGreedy(const std::vector<std::vector<int>> & graph,
                                                       const std::vector<int> & order);

    std::vector<std::vector<int>> partitionGraphDSatur(const std::vector<std::vector<int>> & graph);
}

#endif /* LIBS_GRAPHS_INC_GRAPHS_COLOR */
