#include <mesh/colorMesh.h>

#include <algorithm>
#include <iostream>

#include <graphs/color.h>

#include <utils/stopwatch.h>

namespace mesh
{
    std::vector<std::vector<int>> createMeshElementGraph(const TriangleMesh & mesh)
    {
        const int nNodes = mesh.nodes.size();
        const int nElems = mesh.elements.size();

        // Elements which use node i
        std::vector<std::vector<int>> nodeElements(nNodes);
        for (int i = 0; i < nElems; i++)
        {
            for (int node : mesh.elements[i])
            {
                nodeElements[node].push_back(i);
            }
        }

        // Build the adjacency list
        std::vector<std::vector<int>> graph(nElems);

        for (int i = 0; i < nNodes; i++)
        {
            auto & curr = nodeElements[i];

            // Sort and remove duplicates
            // Not sure if there should be duplicates, but better be safe
            std::sort(curr.begin(), curr.end());
            auto last = std::unique(curr.begin(), curr.end());
            curr.erase(last, curr.end());

            const int m = curr.size();
            for (int j = 0; j < m; j++)
            {
                const int from = curr[j];
                for (int k = j + 1; k < m; k++)
                {
                    const int to = curr[k];
                    graph[from].push_back(to);
                    graph[to].push_back(from);
                }
            }
        }

        return graph;
    }

    std::vector<std::vector<int>> colorMeshElements(const TriangleMesh & mesh)
    {
        u::Stopwatch sw;

        // Create the graph -- elements are connected if they share a node
        auto graph = createMeshElementGraph(mesh);
        const auto tGraph = sw.millis(true);

        const auto order = graphs::buildSmallestLastOrdering(graph);
        const auto tOrder = sw.millis(true);

        const auto coloring = graphs::partitionGraphGreedy(graph, order);
        const auto tColor = sw.millis();

        std::cout << "Mesh coloring times:\n";
        std::cout << "Create graph: " << tGraph << " ms\n";
        std::cout << "Create ordering: " << tOrder << " ms\n";
        std::cout << "Greedy coloring: " << tColor << " ms\n";

        return coloring;
    }
} // namespace mesh