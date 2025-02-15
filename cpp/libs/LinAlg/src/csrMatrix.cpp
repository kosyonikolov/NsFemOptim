#include <linAlg/csrMatrix.h>

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

    template void CsrMatrix<float>::rMult(const float * src, float * dst) const;
    template void CsrMatrix<double>::rMult(const double * src, double * dst) const;
} // namespace linalg