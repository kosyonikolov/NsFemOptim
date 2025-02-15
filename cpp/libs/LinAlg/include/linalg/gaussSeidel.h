#ifndef LIBS_LINALG_INCLUDE_LINALG_GAUSSSEIDEL
#define LIBS_LINALG_INCLUDE_LINALG_GAUSSSEIDEL

#include <linalg/csrMatrix.h>

namespace linalg
{
    template <typename F>
    double gaussSeidel(const CsrMatrix<F> & m, F * x, const F * b,
                       const int maxIters, const double eps);

    template <typename F, VectorLike<F> A, VectorLike<F> B>
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
} // namespace linalg

#endif /* LIBS_LINALG_INCLUDE_LINALG_GAUSSSEIDEL */
