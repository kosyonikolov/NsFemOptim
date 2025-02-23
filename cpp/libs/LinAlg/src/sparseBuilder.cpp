#include <linalg/sparseBuilder.h>

#include <stdexcept>
#include <format>
#include <cassert>
#include <algorithm>

#include <linalg/csrMatrix.h>

namespace linalg
{
    template<typename F>
    SparseMatrixBuilder<F>::SparseMatrixBuilder(const int rows, const int cols) : rows(rows), cols(cols)
    {
        if (rows < 1 || cols < 1)
        {
            throw std::invalid_argument(std::format("Bad builder matrix size: {}x{}", rows, cols));
        }

        rowPairs.resize(rows);
    }

    template<typename F>
    void SparseMatrixBuilder<F>::resize(const int newRows, const int newCols)
    {
        if (newRows < 1 || newCols < 1)
        {
            throw std::invalid_argument(std::format("Bad builder matrix size: {}x{}", newRows, newCols));
        }

        rows = newRows;
        cols = newCols;

        rowPairs.clear();
        rowPairs.resize(newRows);
    }

    template<typename F>
    void SparseMatrixBuilder<F>::add(const int row, const int col, F value)
    {
        assert(row >= 0 && row < rows);
        assert(col >= 0 && col < cols);
        rowPairs[row].emplace_back(col, value);
    }

    template<typename F>
    void SparseMatrixBuilder<F>::compressRows()
    {
        auto cmp = [](const ColPair & a, const ColPair & b)
        {
            return a.col < b.col;
        };

        for (std::vector<ColPair> & row : rowPairs)
        {
            if (row.empty())
            {
                continue;
            }

            std::sort(row.begin(), row.end(), cmp);
            const int n = row.size();
            int u = 0; // unique index
            int i = 1; // source index
            for (; i < n; i++)
            {
                if (row[i].col == row[u].col)
                {
                    row[u].value += row[i].value;
                }
                else
                {
                    // New unique value - copy it to the next position
                    u++;
                    row[u] = row[i];
                }
            }

            row.resize(u + 1);
        }
    }

    template<typename F>
    const std::vector<std::vector<typename SparseMatrixBuilder<F>::ColPair>> & SparseMatrixBuilder<F>::getRows() const
    {
        return rowPairs;
    }

    template<typename F>
    CsrMatrix<F> SparseMatrixBuilder<F>::buildCsr()
    {
        compressRows();
        
        int nnz = 0;
        for (const auto & row : rowPairs)
        {
            nnz += row.size();
        }

        CsrMatrix<F> result;
        result.rows = rows;
        result.cols = cols;
        result.values.resize(nnz);
        result.colIdx.resize(nnz);
        result.rowStart.resize(rows + 1);

        int j = 0; // Index in values/colIdx
        for (int i = 0; i < rows; i++)
        {
            result.rowStart[i] = j;
            const std::vector<ColPair> & row = rowPairs[i];
            for (int k = 0; k < row.size(); k++, j++)
            {
                result.values[j] = row[k].value;
                result.colIdx[j] = row[k].col;
            }
        }
        result.rowStart.back() = nnz;
        
        return result;
    }

    template SparseMatrixBuilder<float>::SparseMatrixBuilder(const int rows, const int cols);
    template void SparseMatrixBuilder<float>::add(const int row, const int col, float value);
    template void SparseMatrixBuilder<float>::compressRows();
    template const std::vector<std::vector<typename SparseMatrixBuilder<float>::ColPair>> & SparseMatrixBuilder<float>::getRows() const;
    template CsrMatrix<float> SparseMatrixBuilder<float>::buildCsr();
}