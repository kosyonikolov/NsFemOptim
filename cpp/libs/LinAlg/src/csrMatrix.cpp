#include <linalg/csrMatrix.h>

#include <cmath>

namespace linalg
{
    template <typename F>
    void CsrMatrix<F>::rMult(const F * src, F * dst) const
    {
        for (int i = 0; i < rows; i++)
        {
            const int j1 = rowStart[i + 1];
            F sum = 0;
            for (int j = rowStart[i]; j < j1; j++)
            {
                const int col = colIdx[j];
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
                const int col = colIdx[j];
                sum += values[j] * x[col];
            }
            const auto delta = sum - b[i];
            errSum += delta * delta;
        }
        return std::sqrt(errSum / rows);
    }

    template void CsrMatrix<float>::rMult(const float * src, float * dst) const;
    template void CsrMatrix<double>::rMult(const double * src, double * dst) const;

    template double CsrMatrix<float>::mse(const float * x, const float * b) const;
    template double CsrMatrix<double>::mse(const double * x, const double * b) const;
} // namespace linalg