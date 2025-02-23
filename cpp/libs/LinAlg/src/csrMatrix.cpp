#include <linalg/csrMatrix.h>

#include <cmath>
#include <stdexcept>
#include <format>
#include <algorithm>

namespace linalg
{
    template<typename F>
    CsrMatrix<F> CsrMatrix<F>::slice(std::span<const int> & rowIds, std::span<int> & colIds) const
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
        result.rowStart.resize(nNewRows);
        
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

        return result;
    }

    template <typename F>
    bool CsrMatrix<F>::operator==(const CsrMatrix<F> & other) const
    {
        if (rows != other.rows)
        {
            return false;
        }

        if (cols != other.cols)
        {
            return false;
        }

        if (values != other.values)
        {
            return false;
        }

        if (column != other.column)
        {
            return false;
        }

        if (rowStart != other.rowStart)
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

    template CsrMatrix<float> CsrMatrix<float>::slice(std::span<const int> & rowIds, std::span<int> & colIds) const;

    template bool CsrMatrix<float>::operator!=(const CsrMatrix<float> & other) const;
    template bool CsrMatrix<double>::operator!=(const CsrMatrix<double> & other) const;

    template void CsrMatrix<float>::rMult(const float * src, float * dst) const;
    template void CsrMatrix<double>::rMult(const double * src, double * dst) const;

    template double CsrMatrix<float>::mse(const float * x, const float * b) const;
    template double CsrMatrix<double>::mse(const double * x, const double * b) const;
} // namespace linalg