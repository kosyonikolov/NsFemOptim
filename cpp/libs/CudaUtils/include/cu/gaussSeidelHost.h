#ifndef LIBS_CUDAUTILS_INCLUDE_CU_GAUSSSEIDELHOST
#define LIBS_CUDAUTILS_INCLUDE_CU_GAUSSSEIDELHOST

#include <memory>

#include <linalg/csrMatrix.h>

namespace cu
{
    class Blas;
    class Sparse;
    class GaussSeidel;

    class GaussSeidelHost
    {
        std::unique_ptr<Blas> blas;
        std::unique_ptr<Sparse> sparse;
        std::unique_ptr<GaussSeidel> cudaGs;

    public:
        GaussSeidelHost(const linalg::CsrMatrix<float> & m);

        ~GaussSeidelHost();

        float solve(const std::vector<float> & rhs, std::vector<float> & sol,
                    const int maxIters, const float target);

        void setMseCheckInterval(const int newInterval);
    };
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_GAUSSSEIDELHOST */
