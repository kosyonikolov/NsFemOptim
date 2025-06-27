#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>
#include <format>

#include <linalg/graphs.h>

namespace linalg
{
    bool verifyColoring(const std::vector<std::vector<int>> & graph, const std::vector<std::vector<int>> & coloring)
    {
        const int n = graph.size();
        std::vector<int> color(n, -1);

        bool ok = true;
        const int nP = coloring.size();
        for (int i = 0; i < nP; i++)
        {
            const auto & part = coloring[i];
            for (int v : part)
            {
                if (v < 0 || v >= n)
                {
                    ok = false;
                    std::cerr << "Partition " << i << " contains vertex " << v << ", which doesn't exist in the graph\n";
                    continue;
                }
                if (color[v] != -1)
                {
                    ok = false;
                    std::cerr << "Partition " << i << " tried to color vertex " << v << " again. It already has color " << color[v] << "\n";
                    continue;
                }
                color[v] = i;
            }
        }

        for (int v = 0; v < n; v++)
        {
            if (color[v] == -1)
            {
                ok = false;
                std::cerr << "Vertex " << v << " is not colored\n";
                continue;
            }

            const int currColor = color[v];
            const auto & neigh = graph[v];
            for (int nv : neigh)
            {
                const int nColor = color[nv];
                if (currColor == nColor)
                {
                    ok = false;
                    std::cerr << std::format("Vertex {} has neighbour {} with the same color [{}]\n", v, nv, currColor);
                }
            }
        }

        return ok;
    }

    std::vector<int> buildSmallestLastOrdering(const std::vector<std::vector<int>> & graph)
    {
        const int n = graph.size();
        std::vector<int> result(n);
        std::vector<int> degree(n);
        std::vector<bool> removed(n, false);

        struct E
        {
            int vertex;
            int degree;
            bool operator<(const E & other) const
            {
                return degree > other.degree; // It's a max heap by default
            }
        };

        std::priority_queue<E> q;

        for (int i = 0; i < n; i++)
        {
            degree[i] = graph[i].size();
            q.push({i, degree[i]});
        }

        for (int i = 0; i < n; i++)
        {
            int v = -1;
            while (true)
            {
                auto curr = q.top();
                q.pop();
                if (removed[curr.vertex])
                {
                    continue;
                }
                v = curr.vertex;
                break;
            }
            assert(v >= 0 && v < n);

            result[i] = v;
            removed[v] = true;
            const auto & neigh = graph[v];
            for (int nv : neigh)
            {
                if (removed[nv])
                {
                    continue;
                }
                degree[nv]--;
                q.push({nv, degree[nv]});
            }
        }

        std::reverse(result.begin(), result.end());
        return result;
    }

    std::vector<std::vector<int>> partitionGraphGreedy(const std::vector<std::vector<int>> & graph,
                                                       const std::vector<int> & order)
    {
        const int n = graph.size();
        assert(n == order.size());
        std::vector<int> color(n, -1);
        std::vector<bool> used;
        std::vector<std::vector<int>> result;

        color[order[0]] = 0;
        result.push_back({order[0]});
        used = {false}; // used.size() == result.size()

        for (int k = 1; k < n; k++)
        {
            const int i = order[k];
            const auto & neigh = graph[i];
            std::fill(used.begin(), used.end(), false);
            for (int j : neigh)
            {
                const int nColor = color[j];
                if (nColor != -1)
                {
                    assert(nColor < used.size());
                    used[nColor] = true;
                }
            }

            int lowest = 0;
            bool found = false;
            for (; lowest < used.size(); lowest++)
            {
                if (!used[lowest])
                {
                    found = true;
                    break;
                }
            }

            if (found)
            {
                color[i] = lowest;
                result[lowest].push_back(i);
            }
            else
            {
                color[i] = lowest;
                used.push_back(false);
                result.push_back({i});
            }
        }

        return result;
    }

    std::vector<std::vector<int>> partitionGraphDSatur(const std::vector<std::vector<int>> & graph)
    {
        const int n = graph.size();
        std::vector<int> degree(n, 0), saturation(n, 0), color(n, -1);
        std::vector<bool> used(n, false);
        std::vector<std::vector<int>> result;
        for (int i = 0; i < n; i++)
        {
            degree[i] = graph[i].size();
        }

        auto selectVertex = [&]() -> int
        {
            int idx = -1;
            int deg = -1;
            int sat = -1;
            for (int i = 0; i < n; i++)
            {
                if (color[i] >= 0)
                {
                    continue;
                }
                if (saturation[i] < sat)
                {
                    continue;
                }
                if (saturation[i] > sat)
                {
                    idx = i;
                    sat = saturation[i];
                    deg = degree[i];
                }
                else if (degree[i] > deg) // Equal saturation
                {
                    idx = i;
                    deg = degree[i];
                }
            }

            return idx;
        };

        auto colorVertex = [&](const int v)
        {
            const int k = result.size(); // Current number of colors
            // std::fill_n(used.begin(), used.begin() + k, false);
            for (int i = 0; i < k; i++)
            {
                used[i] = false;
            }
            for (const int u : graph[v])
            {
                if (color[u] >= 0)
                {
                    used[color[u]] = true;
                }
                else
                {
                    // Current vertex is about to get colored & disconnected
                    degree[u]--;
                    saturation[u]++;
                }
            }
            int newColor = 0;
            while (newColor < n && used[newColor])
            {
                newColor++;
            }
            if (newColor >= k)
            {
                assert(newColor == k);
                result.push_back({});
            }
            color[v] = newColor;
            result[newColor].push_back(v);
        };

        for (int i = 0; i < n; i++)
        {
            const int v = selectVertex();
            colorVertex(v);
        }

        return result;
    }

    template <typename F>
    std::vector<std::vector<int>> buildCsrGraph(const linalg::CsrMatrix<F> & m)
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
                }
            }
        }

        return graph;
    }

    template std::vector<std::vector<int>> buildCsrGraph(const linalg::CsrMatrix<float> & m);
    template std::vector<std::vector<int>> buildCsrGraph(const linalg::CsrMatrix<double> & m);
} // namespace linalg