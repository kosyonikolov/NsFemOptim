#include <linalg/sparseDokBuilder.h>

#include <algorithm>
#include <cassert>

#include <linalg/csrMatrix.h>

namespace linalg
{
    template <typename F>
    SparseMatrixDokBuilder<F>::SparseMatrixDokBuilder(const int rows, const int cols)
        : rows(rows), cols(cols)
    {
    }

    template <typename F>
    void SparseMatrixDokBuilder<F>::resize(const int newRows, const int newCols)
    {
        rows = newRows;
        cols = newCols;
    }

    template <typename F>
    void SparseMatrixDokBuilder<F>::add(const int row, const int col, F value)
    {
        assert(row >= 0 && row < rows);
        assert(col >= 0 && col < cols);
        triplets.emplace_back(row, col, value);
    }

    template <typename F>
    void SparseMatrixDokBuilder<F>::compress()
    {
        if (triplets.empty())
        {
            // Nothing to do
            return;
        }

        auto cmp = [](const Triplet<F> & a, const Triplet<F> & b)
        {
            if (a.row != b.row)
            {
                return a.row < b.row;
            }
            return a.col < b.col;
        };

        std::sort(triplets.begin(), triplets.end(), cmp);

        const int n = triplets.size();
        int j = 0; // Survivor idx
        for (int i = 1; i < n; i++)
        {
            auto & s = triplets[j];
            const auto & curr = triplets[i];
            if (curr.row == s.row && curr.col == s.col)
            {
                s.value += curr.value;
            }
            else
            {
                j++;
                triplets[j] = curr;
            }
        }

        triplets.resize(j + 1);
    }

    template <typename F>
    CsrMatrix<F> SparseMatrixDokBuilder<F>::buildCsr()
    {
        compress();
        const int nnz = triplets.size();

        CsrMatrix<F> result;
        result.rows = rows;
        result.cols = cols;
        result.values.resize(nnz);
        result.column.resize(nnz);
        result.rowStart.resize(rows + 1, nnz);

        if (nnz == 0)
        {
            return result;
        }

        int row = -1;
        for (int i = 0; i < nnz; i++)
        {
            const Triplet<F> & curr = triplets[i];
            if (curr.row != row)
            {
                row = curr.row;
                result.rowStart[row] = i;
            }
            result.column[i] = curr.col;
            result.values[i] = curr.value;
        }

        return result;
    }

    template <typename F>
    int SparseMatrixDokBuilder<F>::numRows() const
    {
        return rows;
    }

    template <typename F>
    int SparseMatrixDokBuilder<F>::numCols() const
    {
        return cols;
    }

    template SparseMatrixDokBuilder<float>::SparseMatrixDokBuilder(const int rows, const int cols);
    template void SparseMatrixDokBuilder<float>::resize(const int newRows, const int newCols);
    template void SparseMatrixDokBuilder<float>::add(const int row, const int col, float value);
    template void SparseMatrixDokBuilder<float>::compress();
    template CsrMatrix<float> SparseMatrixDokBuilder<float>::buildCsr();
    template int SparseMatrixDokBuilder<float>::numRows() const;
    template int SparseMatrixDokBuilder<float>::numCols() const;
} // namespace linalg