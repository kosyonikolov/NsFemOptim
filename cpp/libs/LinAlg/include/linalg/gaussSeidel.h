#ifndef LIBS_LINALG_INCLUDE_LINALG_GAUSSSEIDEL
#define LIBS_LINALG_INCLUDE_LINALG_GAUSSSEIDEL

#include <linalg/csrMatrix.h>

namespace linalg
{
    template <typename F>
    struct GaussSeidelContext
    {
        CsrMatrix<F> stripped; // Matrix without diagonal elements
        std::vector<F> invDiag; // 1.0 / diag
    };

    template<typename F>
    GaussSeidelContext<F> buildGaussSeidelContext(const CsrMatrix<F> & m); 

    template <typename F>
    double gaussSeidel(const CsrMatrix<F> & m, F * x, const F * b,
                       const int maxIters, const double eps);

    template <typename F, u::VectorLike<F> A, u::VectorLike<F> B>
    double gaussSeidel(const CsrMatrix<F> & m, A & x, const B & b,
                       const int maxIters, const double eps)
    {
        if (x.size() != m.cols)
        {
            throw std::invalid_argument(std::format("{}: Bad size of x vector [{}] - expected {}",
                                                    __FUNCTION__, x.size(), m.cols));
        }
        if (b.size() != m.rows)
        {
            throw std::invalid_argument(std::format("{}: Bad size of b vector [{}] - expected {}",
                                                    __FUNCTION__, b.size(), m.rows));
        }
        return gaussSeidel(m, x.data(), b.data(), maxIters, eps);
    }

    template <typename F>
    double gaussSeidelCustomOrder(const CsrMatrix<F> & m, F * x, const F * b,
                                  const int * order,
                                  const int maxIters, const double eps);

    template <typename F, u::VectorLike<F> A, u::VectorLike<F> B, u::VectorLike<int> C>
    double gaussSeidelCustomOrder(const CsrMatrix<F> & m, A & x, const B & b,
                                  const  C & order,
                                  const int maxIters, const double eps)
    {
        if (x.size() != m.cols)
        {
            throw std::invalid_argument(std::format("{}: Bad size of x vector [{}] - expected {}",
                                                    __FUNCTION__, x.size(), m.cols));
        }
        if (b.size() != m.rows)
        {
            throw std::invalid_argument(std::format("{}: Bad size of b vector [{}] - expected {}",
                                                    __FUNCTION__, b.size(), m.rows));
        }
        if (order.size() != x.size())
        {
            throw std::invalid_argument(std::format("{}: Bad size of order vector [{}] - expected {}",
                                                    __FUNCTION__, order.size(), x.size()));
        }
        return gaussSeidelCustomOrder(m, x.data(), b.data(), order.data(), maxIters, eps);
    }

    template <typename F>
    void mse2ch(const CsrMatrix<F> & m, F * x, const F * b, double & mse0, double & mse1);

    // x and b are both matrices with two columns, in row-major order (interleaved channels)
    template <typename F>
    std::tuple<double, double> gaussSeidel2ch(const CsrMatrix<F> & m, F * x, const F * b,
                                              const int maxIters, const double eps);

    template <typename F, u::VectorLike<F> A, u::VectorLike<F> B>
    std::tuple<double, double> gaussSeidel2ch(const CsrMatrix<F> & m, A & x, const B & b,
                                              const int maxIters, const double eps)
    {
        if (x.size() != 2 * m.cols)
        {
            throw std::invalid_argument(std::format("{}: Bad size of x vector [{}] - expected {}",
                                                    __FUNCTION__, x.size(), 2 * m.cols));
        }
        if (b.size() != 2 * m.rows)
        {
            throw std::invalid_argument(std::format("{}: Bad size of b vector [{}] - expected {}",
                                                    __FUNCTION__, b.size(), 2 * m.rows));
        }
        return gaussSeidel2ch(m, x.data(), b.data(), maxIters, eps);
    }

} // namespace linalg

#endif /* LIBS_LINALG_INCLUDE_LINALG_GAUSSSEIDEL */
