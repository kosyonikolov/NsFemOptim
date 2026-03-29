#include <cassert>

#include <linalg/graphs.h>

namespace linalg
{
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