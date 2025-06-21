#include <linalg/csrMatrix.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <stdexcept>

namespace linalg
{
    template <typename F>
    CsrMatrix<F> CsrMatrix<F>::slice(std::span<const int> rowIds, std::span<const int> colIds) const
    {
        if (rowIds.empty() || colIds.empty())
        {
            CsrMatrix<F> result;
            result.rows = 0;
            result.cols = 0;
            return result;
        }

        // Sanity check
        for (int i : rowIds)
        {
            if (i < 0 || i >= rows)
            {
                throw std::invalid_argument(std::format("{}: Bad row index {}", __FUNCTION__, i));
            }
        }

        // dstId = colMap[srcId]
        // -1 = not used
        std::vector<int> colMap(cols, -1);
        for (int i = 0; i < colIds.size(); i++)
        {
            const int j = colIds[i];
            if (j < 0 || j >= cols)
            {
                throw std::invalid_argument(std::format("{}: Bad col index {}", __FUNCTION__, j));
            }
            colMap[j] = i;
        }

        struct Pair
        {
            int id;
            F value;
            bool operator<(const Pair & other) const
            {
                return id < other.id;
            }
        };

        std::vector<Pair> tmpRow;
        const int nNewRows = rowIds.size();
        const int nNewCols = colIds.size();
        CsrMatrix<F> result;
        result.rows = nNewRows;
        result.cols = nNewCols;
        result.rowStart.resize(nNewRows + 1);

        for (int i = 0; i < nNewRows; i++)
        {
            result.rowStart[i] = result.values.size();

            tmpRow.clear();
            const int srcRow = rowIds[i];
            const int j1 = rowStart[srcRow + 1];
            for (int j = rowStart[srcRow]; j < j1; j++)
            {
                const int srcCol = column[j];
                const int dstCol = colMap[srcCol];
                if (dstCol < 0)
                {
                    continue;
                }
                tmpRow.emplace_back(dstCol, values[j]);
            }

            std::sort(tmpRow.begin(), tmpRow.end());
            for (const Pair & p : tmpRow)
            {
                result.column.push_back(p.id);
                result.values.push_back(p.value);
            }
        }

        result.rowStart.back() = result.values.size();

        return result;
    }

    template <typename F>
    void CsrMatrix<F>::findOffsets(const int row, std::span<const int> columnIds, std::span<int> dstOffsets) const
    {
        assert(std::is_sorted(columnIds.begin(), columnIds.end()));
        assert(columnIds.size() == dstOffsets.size());

        if (row < 0 || row >= rows)
        {
            std::fill(dstOffsets.begin(), dstOffsets.end(), -1);
            return;
        }

        int j = 0;
        const int j1 = rowStart[row + 1];
        int i = 0;
        const int i1 = columnIds.size();
        while (i < i1 && j < j1)
        {
            if (columnIds[i] == column[j])
            {
                dstOffsets[i] = j;
            }
            else if (columnIds[i] > column[j])
            {
                j++;
            }
            else // columnIds[i] < column[j]
            {
                dstOffsets[i] = -1;
                i++;
            }
        }

        while (i < i1)
        {
            dstOffsets[i] = -1;
            i++;
        }
    }

    template <typename F>
    void CsrMatrix<F>::findOffsetsUnsorted(const int row, std::span<const int> columnIds, std::span<int> dstOffsets) const
    {
        assert(columnIds.size() == dstOffsets.size());

        if (row < 0 || row >= rows)
        {
            std::fill(dstOffsets.begin(), dstOffsets.end(), -1);
            return;
        }

        const int * pStart = column.data() + rowStart[row];
        const int * pEnd = column.data() + rowStart[row + 1];
        for (int i = 0; i < columnIds.size(); i++)
        {
            const int q = columnIds[i];
            const int * firstGeq = std::lower_bound(pStart, pEnd, q);
            if (firstGeq == pEnd || *firstGeq != q)
            {
                dstOffsets[i] = -1;
            } 
            else
            {
                dstOffsets[i] = firstGeq - column.data();
            }
        }
    }

    template <typename F>
    bool CsrMatrix<F>::compareLayout(const CsrMatrix<F> & other) const
    {
        if (rows != other.rows)
        {
            return false;
        }

        if (cols != other.cols)
        {
            return false;
        }

        if (rowStart != other.rowStart)
        {
            return false;
        }

        if (column != other.column)
        {
            return false;
        }

        return true;
    }

    template <typename F>
    bool CsrMatrix<F>::compareValues(const CsrMatrix<F> & other, const F epsilon) const
    {
        if (values.size() != other.values.size())
        {
            return false;
        }

        const size_t n = values.size();
        for (size_t i = 0; i < n; i++)
        {
            const F delta = values[i] - other.values[i];
            if (std::abs(delta) > epsilon)
            {
                return false;
            }
        }

        return true;
    }

    template <typename F>
    bool CsrMatrix<F>::operator==(const CsrMatrix<F> & other) const
    {
        if (!compareLayout(other))
        {
            return false;
        }

        if (!compareValues(other, 0))
        {
            return false;
        }

        return true;
    }

    template <typename F>
    bool CsrMatrix<F>::operator!=(const CsrMatrix<F> & other) const
    {
        return !operator==(other);
    }

    template <typename F>
    void CsrMatrix<F>::rMult(const F * src, F * dst) const
    {
        for (int i = 0; i < rows; i++)
        {
            const int j1 = rowStart[i + 1];
            F sum = 0;
            for (int j = rowStart[i]; j < j1; j++)
            {
                const int col = column[j];
                sum += values[j] * src[col];
            }
            dst[i] = sum;
        }
    }

    template <typename F>
    void CsrMatrix<F>::rMultD(const double * src, double * dst) const
    {
        for (int i = 0; i < rows; i++)
        {
            const int j1 = rowStart[i + 1];
            double sum = 0;
            for (int j = rowStart[i]; j < j1; j++)
            {
                const int col = column[j];
                sum += values[j] * src[col];
            }
            dst[i] = sum;
        }
    }

    template <typename F>
    double CsrMatrix<F>::mse(const F * x, const F * b) const
    {
        double errSum = 0;
        for (int i = 0; i < rows; i++)
        {
            const int j1 = rowStart[i + 1];
            F sum = 0;
            for (int j = rowStart[i]; j < j1; j++)
            {
                const int col = column[j];
                sum += values[j] * x[col];
            }
            const auto delta = sum - b[i];
            errSum += delta * delta;
        }
        return std::sqrt(errSum / rows);
    }

    template CsrMatrix<float> CsrMatrix<float>::slice(std::span<const int> rowIds, std::span<const int> colIds) const;

    template void CsrMatrix<float>::findOffsets(const int row, std::span<const int> columnIds, std::span<int> dstOffsets) const;

    template void CsrMatrix<float>::findOffsetsUnsorted(const int row, std::span<const int> columnIds, std::span<int> dstOffsets) const;

    template bool CsrMatrix<float>::compareLayout(const CsrMatrix<float> & other) const;
    template bool CsrMatrix<double>::compareLayout(const CsrMatrix<double> & other) const;

    template bool CsrMatrix<float>::compareValues(const CsrMatrix<float> & other, const float epsilon) const;
    template bool CsrMatrix<double>::compareValues(const CsrMatrix<double> & other, const double epsilon) const;

    template bool CsrMatrix<float>::operator!=(const CsrMatrix<float> & other) const;
    template bool CsrMatrix<double>::operator!=(const CsrMatrix<double> & other) const;

    template void CsrMatrix<float>::rMult(const float * src, float * dst) const;
    template void CsrMatrix<double>::rMult(const double * src, double * dst) const;

    template void CsrMatrix<float>::rMultD(const double * src, double * dst) const;
    template void CsrMatrix<double>::rMultD(const double * src, double * dst) const;

    template double CsrMatrix<float>::mse(const float * x, const float * b) const;
    template double CsrMatrix<double>::mse(const double * x, const double * b) const;
} // namespace linalg