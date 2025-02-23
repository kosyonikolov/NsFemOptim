#include <linalg/eigen.h>

#include <algorithm>
#include <format>
#include <stdexcept>

namespace linalg
{
    template <typename F>
    CsrMatrix<F> csrFromEigen(const Eigen::SparseMatrix<F, Eigen::RowMajor> & m)
    {
        if (!m.isCompressed())
        {
            throw std::invalid_argument(std::format("{}: Eigen matrix must be compressed", __FUNCTION__));
        }

        const int nRows = m.rows();
        const int nCols = m.cols();
        const int nnz = m.nonZeros();

        const auto valPtr = m.valuePtr();
        const auto colPtr = m.innerIndexPtr();
        const auto rowStartPtr = m.outerIndexPtr();

        CsrMatrix<F> result;
        result.rows = nRows;
        result.cols = nCols;
        result.values.resize(nnz);
        result.column.resize(nnz);
        result.rowStart.resize(nRows + 1);

        std::copy_n(valPtr, nnz, result.values.data());
        std::copy_n(colPtr, nnz, result.column.data());
        std::copy_n(rowStartPtr, nRows, result.rowStart.data());
        result.rowStart.back() = nnz; // Sentinel value

        return result;
    }

    template <typename F>
    Eigen::SparseMatrix<F, Eigen::RowMajor> eigenFromCsr(const CsrMatrix<F> & m)
    {
        // TODO
    }

    template <typename F>
    Eigen::SparseMatrix<F, Eigen::RowMajor> buildEigenCsr(SparseMatrixBuilder<F> & builder)
    {
        builder.compressRows();
        const auto rowPairs = builder.getRows();
        using Triplet = Eigen::Triplet<F>;

        std::vector<Triplet> triplets;
        const int nRows = rowPairs.size();
        for (int i = 0; i < nRows; i++)
        {
            for (const auto & pair : rowPairs[i])
            {
                triplets.emplace(i, pair.col, pair.value);
            }
        }

        Eigen::SparseMatrix<F, Eigen::RowMajor> result(nRows, builder.numCols());
        result.setFromTriplets(triplets);
        return result;
    }

    template CsrMatrix<float> csrFromEigen(const Eigen::SparseMatrix<float, Eigen::RowMajor> & m);
    template CsrMatrix<double> csrFromEigen(const Eigen::SparseMatrix<double, Eigen::RowMajor> & m);
} // namespace linalg