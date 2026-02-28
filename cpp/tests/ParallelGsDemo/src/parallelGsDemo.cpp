#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

#include <linalg/csrMatrix.h>
#include <linalg/graphs.h>

template <typename F>
std::vector<std::vector<int>> buildCsrGraphGeneral(const linalg::CsrMatrix<F> & m)
{
    const int n = m.rows;

    std::vector<std::vector<int>> graph(n);
    for (int r = 0; r < n; r++)
    {
        const int j1 = m.rowStart[r + 1];
        for (int j = m.rowStart[r]; j < j1; j++)
        {
            const int c = m.column[j];
            if (r != c)
            {
                graph[r].push_back(c);
                graph[c].push_back(r);
            }
        }
    }

    return graph;
}

template <typename F>
linalg::CsrMatrix<F> sparsify(const F * data, const int rows, const int cols)
{
    linalg::CsrMatrix<F> result;
    result.rows = rows;
    result.cols = cols;
    for (int r = 0; r < rows; r++)
    {
        result.rowStart.push_back(result.values.size());
        const F * row = data + r * cols;
        for (int c = 0; c < cols; c++)
        {
            if (row[c] != 0)
            {
                result.values.push_back(row[c]);
                result.column.push_back(c);
            }
        }
    }
    result.rowStart.push_back(result.values.size());
    return result;
}

void printGraphviz(const std::vector<std::vector<int>> & graph, const std::vector<std::vector<int>> & colors)
{
    struct Edge
    {
        int from, to;
        bool operator<(const Edge & other) const
        {
            if (from == other.from)
            {
                return to < other.to;
            }
            return from < other.from;
        }

        bool operator==(const Edge & other) const
        {
            return from == other.from && to == other.to;
        }
    };

    const int n = graph.size();
    std::vector<Edge> edges;
    for (int i = 0; i < n; i++)
    {
        for (int j : graph[i])
        {
            Edge p;
            p.from = std::min(i, j);
            p.to = std::max(i, j);
            edges.push_back(p);
        }
    }

    // Leave only unique edges
    std::sort(edges.begin(), edges.end());
    auto last = std::unique(edges.begin(), edges.end());
    edges.erase(last, edges.end());

    // Color names for partitions
    std::vector<std::string> colorNames = {"red", "green", "blue"};
    const int nColors = std::min(colorNames.size(), colors.size());
    
    std::cout << "graph G\n{\n";
    for (const auto & e : edges)
    {
        std::cout << "\t" << e.from << " -- " << e.to << "\n";
    }
    for (int p = 0; p < nColors; p++)
    {
        for (int v : colors[p])
        {
            std::cout << "\t" << v << " [color=\"" << colorNames[p] << "\"]\n";
        }
    }
    std::cout << "}\n";
}

int main()
{
    constexpr int n = 7;
    // clang-format off
    std::array<int, n * n> denseMat =
    {
        1, 2, 0, 0, 0, 0, 0,
        3, 4, 5, 0, 0, 0, 0,
        0, 6, 7, 8, 0, 0, 0,
        0, 0, 0, 9, 10, 11, 0,
        0, 0, 0, 0, 11, 12, 13,
        0, 0, 0, 0, 0, 14, 15,
        0, 0, 0, 0, 0, 0, 16
    };
    // clang-format on

    auto csr = sparsify(denseMat.data(), n, n);
    std::cout << "Original matrix:\n"
              << csr << "\n";

    auto graph = buildCsrGraphGeneral(csr);
    auto slOrder = linalg::buildSmallestLastOrdering(graph);
    auto parts = linalg::partitionGraphGreedy(graph, slOrder);
    std::cout << "Graph of matrix:\n";
    printGraphviz(graph, parts);

    const int nColors = parts.size();
    std::cout << "Number of partitions: " << nColors << "\n";
    std::cout << "Original partitions:\n";
    for (auto & v : parts)
    {
        for (int k : v)
        {
            std::cout << k << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Sort the individual partitions and place them in the coloring vector
    std::vector<int> coloring, partitionStart;
    coloring.resize(n);
    partitionStart.resize(nColors + 1);
    partitionStart.back() = n;
    int i = 0;
    for (int c = 0; c < nColors; c++)
    {
        auto & p = parts[c];
        std::sort(p.begin(), p.end());
        std::copy_n(p.begin(), p.size(), coloring.begin() + i);
        partitionStart[c] = i;
        i += p.size();
    }

    // Reorder the matrix to make the coloring redundant -
    // first partition is [0, P1), second is [P1, P2) and so on
    auto reordered = csr.slice(coloring, coloring);

    std::cout << "Reordered matrix:\n"
              << reordered << "\n";

    std::cout << "Partition start:";
    for (int k : partitionStart)
    {
        std::cout << " " << k;
    }
    std::cout << "\n";

    return 0;
}