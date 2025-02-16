#ifndef LIBS_LINALG_INCLUDE_LINALG_GRAPHS
#define LIBS_LINALG_INCLUDE_LINALG_GRAPHS

#include <vector>

#include <linalg/csrMatrix.h>

namespace linalg
{
    std::vector<std::vector<int>> partitionGraphDSatur(const std::vector<std::vector<int>> & graph);

    template <typename F>
    std::vector<std::vector<int>> buildCsrGraph(const linalg::CsrMatrix<F> & m);
} // namespace linalg

#endif /* LIBS_LINALG_INCLUDE_LINALG_GRAPHS */
