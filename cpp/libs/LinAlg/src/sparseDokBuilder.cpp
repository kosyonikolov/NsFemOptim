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
    CsrMatrix<F> SparseMatrixDokBuilder<F>::buildCsr2()
    {
        struct Pair
        {
            int col;
            F value;
            bool operator<(const Pair & other) const
            {
                return col < other.col;
            }
        };

        // Bucket sort
        std::vector<std::vector<Pair>> rowPairs(rows);
        for (const Triplet<F> & t : triplets)
        {
            assert(t.row >= 0 && t.row < rows);
            rowPairs[t.row].emplace_back(t.col, t.value);
        }

        int nnz = 0;
        for (int r = 0; r < rows; r++)
        {
            std::vector<Pair> & rp = rowPairs[r];
            if (rp.empty())
            {
                continue;
            }
            std::sort(rp.begin(), rp.end());
            int j = 0;
            const int n = rp.size();
            for (int i = 1; i < n; i++)
            {
                if (rp[i].col == rp[j].col)
                {
                    rp[j].value += rp[i].value;
                }
                else
                {
                    j++;
                    rp[j] = rp[i];
                }
            }

            j++; // Now it's the NNZ on this row
            rp.resize(j);
            nnz += j;
        }

        CsrMatrix<F> result;
        result.rows = rows;
        result.cols = cols;
        result.values.resize(nnz);
        result.column.resize(nnz);
        result.rowStart.resize(rows + 1, nnz);

        int i = 0;
        for (int r = 0; r < rows; r++)
        {
            const std::vector<Pair> & rp = rowPairs[r];
            result.rowStart[r] = i;
            for (const Pair & p : rp)
            {
                result.values[i] = p.value;
                result.column[i] = p.col;
                i++;
            }
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
    template CsrMatrix<float> SparseMatrixDokBuilder<float>::buildCsr2();
    template int SparseMatrixDokBuilder<float>::numRows() const;
    template int SparseMatrixDokBuilder<float>::numCols() const;
} // namespace linalg