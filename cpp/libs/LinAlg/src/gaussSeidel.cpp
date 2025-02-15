#include <linalg/gaussSeidel.h>

#include <iostream>

namespace linalg
{
    template <typename F>
    void gaussSeidelStep(const CsrMatrix<F> & m, F * x, const F * b)
    {
        const int nRows = m.rows;
        for (int i = 0; i < nRows; i++)
        {
            const int j1 = m.rowStart[i + 1];
            F diag = 0; // element at (i, i)
            double negSum = 0;
            for (int j = m.rowStart[i]; j < j1; j++)
            {
                const int col = m.colIdx[j];
                if (col == i)
                {
                    diag = m.values[j];
                }
                else
                {
                    negSum += m.values[j] * x[col];
                }
            }
            x[i] = (b[i] - negSum) / diag;
        }
    }

    template <typename F>
    double gaussSeidel(const CsrMatrix<F> & m, F * x, const F * b, const int maxIters, const double eps)
    {
        double lastRes = 0;
        for (int i = 0; i < maxIters; i++)
        {
            gaussSeidelStep(m, x, b);
            lastRes = m.mse(x, b);
            std::cout << std::format("{}: {}\n", i, lastRes);

            if (lastRes < eps)
            {
                break;
            }
        }
        return lastRes;
    }

    template double gaussSeidel(const CsrMatrix<float> & m, float * x, const float * b, const int maxIters, const double eps);
    template double gaussSeidel(const CsrMatrix<double> & m, double * x, const double * b, const int maxIters, const double eps);
} // namespace linalg