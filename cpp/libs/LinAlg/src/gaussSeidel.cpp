#include <linalg/gaussSeidel.h>

#include <cmath>
// #include <iostream>

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
                const int col = m.column[j];
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
    void gaussSeidelStep2ch(const CsrMatrix<F> & m, F * x, const F * b)
    {
        const int nRows = m.rows;
        for (int i = 0; i < nRows; i++)
        {
            const int j1 = m.rowStart[i + 1];
            F diag = 0; // element at (i, i)
            double negSum0 = 0;
            double negSum1 = 0;
            for (int j = m.rowStart[i]; j < j1; j++)
            {
                const int col = m.column[j];
                if (col == i)
                {
                    diag = m.values[j];
                }
                else
                {
                    negSum0 += m.values[j] * x[2 * col + 0];
                    negSum1 += m.values[j] * x[2 * col + 1];
                }
            }

            x[2 * i + 0] = (b[2 * i + 0] - negSum0) / diag;
            x[2 * i + 1] = (b[2 * i + 1] - negSum1) / diag;
        }
    }

    template <typename F>
    void mse2ch(const CsrMatrix<F> & m, F * x, const F * b, double & mse0, double & mse1)
    {
        const int rows = m.rows;
        std::array<double, 2> errSum = {0, 0};
        for (int i = 0; i < rows; i++)
        {
            const int j1 = m.rowStart[i + 1];
            std::array<F, 2> sum = {0, 0};
            for (int j = m.rowStart[i]; j < j1; j++)
            {
                const int col = m.column[j];
                sum[0] += m.values[j] * x[2 * col + 0];
                sum[1] += m.values[j] * x[2 * col + 1];
            }
            const auto delta0 = sum[0] - b[2 * i + 0];
            const auto delta1 = sum[1] - b[2 * i + 1];
            errSum[0] += delta0 * delta0;
            errSum[1] += delta1 * delta1;
        }

        mse0 = std::sqrt(errSum[0] / rows);
        mse1 = std::sqrt(errSum[1] / rows);
    }

    template <typename F>
    double gaussSeidel(const CsrMatrix<F> & m, F * x, const F * b,
                       const int maxIters, const double eps)
    {
        double lastRes = -1;
        for (int i = 0; i < maxIters; i++)
        {
            gaussSeidelStep(m, x, b);

            lastRes = m.mse(x, b);
            // std::cout << std::format("{}: {}\n", i, lastRes);

            if (lastRes < eps)
            {
                break;
            }
        }
        return lastRes;
    }

    template <typename F>
    std::tuple<double, double> gaussSeidel2ch(const CsrMatrix<F> & m, F * x, const F * b,
                                              const int maxIters, const double eps)
    {
        double lastRes0 = -1;
        double lastRes1 = -1;
        for (int i = 0; i < maxIters; i++)
        {
            gaussSeidelStep2ch(m, x, b);
            mse2ch(m, x, b, lastRes0, lastRes1);
            if (lastRes0 < eps && lastRes1 < eps)
            {
                break;
            }
        }
        return {lastRes0, lastRes1};
    }

    template double gaussSeidel(const CsrMatrix<float> & m, float * x, const float * b, const int maxIters, const double eps);
    template double gaussSeidel(const CsrMatrix<double> & m, double * x, const double * b, const int maxIters, const double eps);

    template void mse2ch(const CsrMatrix<float> & m, float * x, const float * b, double & mse0, double & mse1);
    template void mse2ch(const CsrMatrix<double> & m, double * x, const double * b, double & mse0, double & mse1);

    template std::tuple<double, double> gaussSeidel2ch(const CsrMatrix<float> & m, float * x, const float * b,
                                                       const int maxIters, const double eps);

    template std::tuple<double, double> gaussSeidel2ch(const CsrMatrix<double> & m, double * x, const double * b,
                                                       const int maxIters, const double eps);
} // namespace linalg