#include <linalg/jacobi.h>

#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace linalg
{
    void jacobiStep(const CsrMatrix<float> & m, float * curr, const float * b, const float * old)
    {
        constexpr float w = 1.0;
        constexpr float cw = 1.0f - w;

        const int nRows = m.rows;
        for (int i = 0; i < nRows; i++)
        {
            const int j1 = m.rowStart[i + 1];
            float diag = 0; // element at (i, i)
            double negSum = 0;
            for (int j = m.rowStart[i]; j < j1; j++)
            {
                const int col = m.column[j];
                if (col == i)
                {
                    diag = m.values[j];
                }
                else
                {
                    negSum += m.values[j] * old[col];
                }
            }
            // curr[i] = (b[i] - negSum) / diag;
            const float newVal = (b[i] - negSum) / diag;
            curr[i] = cw * curr[i] + w * newVal;
        }
    }

    double jacobi(const CsrMatrix<float> & m, float * x, const float * b,
                  const int maxIters, const double eps,
                  float * aux)
    {
        const int n = m.rows;

        std::copy_n(x, n, aux);
        float * curr = aux; // The vector being updated
        float * old = x;
    
        double lastRes = -1;
        for (int i = 0; i < maxIters; i++)
        {
            std::swap(curr, old);
            jacobiStep(m, curr, b, old);

            lastRes = m.mse(curr, b);
            std::cout << std::format("{}: {}\n", i, lastRes);

            if (lastRes < eps)
            {
                break;
            }
        }

        if (curr != x)
        {
            std::copy_n(curr, n, x);
        }

        return lastRes;
    }
} // namespace linalg