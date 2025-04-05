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
    void SparseMatrixDokBuilder<F>::add(std::span<const SparseMatrixDokBuilder<F> *> others)
    {
        // Count extra space
        int extraSpace = 0;
        for (const SparseMatrixDokBuilder<F> * b : others)
        {
            extraSpace += b->triplets.size();
        }
        triplets.reserve(triplets.size() + extraSpace);

        for (const SparseMatrixDokBuilder<F> * b : others)
        {
            triplets.insert(triplets.end(), b->triplets.begin(), b->triplets.end());
        }
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

    SparseMatrixPrototypeBuilder::SparseMatrixPrototypeBuilder(const int rows, const int cols)
        : rows(rows), cols(cols)
    {
    }

    void SparseMatrixPrototypeBuilder::resize(const int newRows, const int newCols)
    {
        rows = newRows;
        cols = newCols;
    }

    void SparseMatrixPrototypeBuilder::add(const int row, const int col)
    {
        assert(row >= 0 && row < rows);
        assert(col >= 0 && col < cols);
        coords.emplace_back(row, col);
    }

    void SparseMatrixPrototypeBuilder::add(std::span<const SparseMatrixPrototypeBuilder *> others)
    {
        // Count extra space
        int extraSpace = 0;
        for (const auto * b : others)
        {
            extraSpace += b->coords.size();
        }
        coords.reserve(coords.size() + extraSpace);

        for (const auto * b : others)
        {
            coords.insert(coords.end(), b->coords.begin(), b->coords.end());
        }
    }

    void SparseMatrixPrototypeBuilder::compress()
    {
        if (coords.empty())
        {
            // Nothing to do
            return;
        }

        auto cmp = [](const Coordinate & a, const Coordinate & b)
        {
            if (a.row != b.row)
            {
                return a.row < b.row;
            }
            return a.col < b.col;
        };

        std::sort(coords.begin(), coords.end(), cmp);

        const int n = coords.size();
        int j = 0; // Survivor idx
        for (int i = 1; i < n; i++)
        {
            auto & s = coords[j];
            const auto & curr = coords[i];
            if (curr.row != s.row || curr.col != s.col)
            {
                j++;
                coords[j] = curr;
            }
        }

        coords.resize(j + 1);
    }

    template <typename F>
    CsrMatrix<F> SparseMatrixPrototypeBuilder::buildCsrPrototype()
    {
        compress();
        const int nnz = coords.size();

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
            const auto & curr = coords[i];
            if (curr.row != row)
            {
                row = curr.row;
                result.rowStart[row] = i;
            }
            result.column[i] = curr.col;
        }

        return result;
    }

    template <typename F>
    CsrMatrix<F> SparseMatrixPrototypeBuilder::buildCsrPrototype2()
    {
        // Bucket sort
        std::vector<std::vector<int>> rowPairs(rows);
        for (const Coordinate & c : coords)
        {
            assert(c.row >= 0 && c.row < rows);
            rowPairs[c.row].emplace_back(c.col);
        }

        int nnz = 0;
        for (int r = 0; r < rows; r++)
        {
            auto & rp = rowPairs[r];
            if (rp.empty())
            {
                continue;
            }
            std::sort(rp.begin(), rp.end());
            int j = 0;
            const int n = rp.size();
            for (int i = 1; i < n; i++)
            {
                if (rp[i] != rp[j])
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
        result.column.resize(nnz);
        result.rowStart.resize(rows + 1, nnz);

        int i = 0;
        for (int r = 0; r < rows; r++)
        {
            const auto & rp = rowPairs[r];
            result.rowStart[r] = i;
            for (const int col : rp)
            {
                result.column[i] = col;
                i++;
            }
        }

        return result;
    }

    int SparseMatrixPrototypeBuilder::numRows() const
    {
        return rows;
    }

    int SparseMatrixPrototypeBuilder::numCols() const
    {
        return cols;
    }

    template SparseMatrixDokBuilder<float>::SparseMatrixDokBuilder(const int rows, const int cols);
    template void SparseMatrixDokBuilder<float>::resize(const int newRows, const int newCols);
    template void SparseMatrixDokBuilder<float>::add(const int row, const int col, float value);
    template void SparseMatrixDokBuilder<float>::add(std::span<const SparseMatrixDokBuilder<float> *> others);
    template void SparseMatrixDokBuilder<float>::compress();
    template CsrMatrix<float> SparseMatrixDokBuilder<float>::buildCsr();
    template CsrMatrix<float> SparseMatrixDokBuilder<float>::buildCsr2();
    template int SparseMatrixDokBuilder<float>::numRows() const;
    template int SparseMatrixDokBuilder<float>::numCols() const;

    template CsrMatrix<float> SparseMatrixPrototypeBuilder::buildCsrPrototype<float>();
    template CsrMatrix<float> SparseMatrixPrototypeBuilder::buildCsrPrototype2<float>();
} // namespace linalg