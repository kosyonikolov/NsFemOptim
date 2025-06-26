#ifndef LIBS_LINALG_INCLUDE_LINALG_JACOBI
#define LIBS_LINALG_INCLUDE_LINALG_JACOBI

#include <linalg/csrMatrix.h>

namespace linalg
{
    double jacobi(const CsrMatrix<float> & m, float * x, const float * b,
                  const int maxIters, const double eps,
                  float * aux);

    template <typename V>
    double jacobi(const CsrMatrix<float> & m, V & x, const V & b,
                  const int maxIters, const double eps,
                  V & aux)
    {
        // TODO Checks
        return jacobi(m, x.data(), b.data(),
                      maxIters, eps,
                      aux.data());
    }
} // namespace linalg

#endif /* LIBS_LINALG_INCLUDE_LINALG_JACOBI */
