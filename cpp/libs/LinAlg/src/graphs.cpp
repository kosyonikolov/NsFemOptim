#include <cassert>

#include <linalg/graphs.h>

namespace linalg
{
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