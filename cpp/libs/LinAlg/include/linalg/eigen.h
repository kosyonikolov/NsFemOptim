#ifndef LIBS_LINALG_INCLUDE_LINALG_EIGEN
#define LIBS_LINALG_INCLUDE_LINALG_EIGEN

// Functions for Eigen interopability

#include <Eigen/Sparse>

#include <linalg/csrMatrix.h>
#include <linalg/sparseBuilder.h>

namespace linalg
{
    template <typename F>
    CsrMatrix<F> csrFromEigen(const Eigen::SparseMatrix<F, Eigen::RowMajor> & m);

    template <typename F>
    Eigen::SparseMatrix<F, Eigen::RowMajor> eigenFromCsr(const CsrMatrix<F> & m);

    template <typename F>
    Eigen::SparseMatrix<F, Eigen::RowMajor> buildEigenCsr(SparseMatrixBuilder<F> & builder);
}

#endif /* LIBS_LINALG_INCLUDE_LINALG_EIGEN */
