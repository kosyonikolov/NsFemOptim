#ifndef LIBS_LINALG_INCLUDE_LINALG_EIGEN
#define LIBS_LINALG_INCLUDE_LINALG_EIGEN

// Functions for Eigen interopability

#include <Eigen/Sparse>

#include <linAlg/csrMatrix.h>

namespace linalg
{
    template <typename F>
    CsrMatrix<F> csrFromEigen(const Eigen::SparseMatrix<F, Eigen::RowMajor> & m);
}

#endif /* LIBS_LINALG_INCLUDE_LINALG_EIGEN */
