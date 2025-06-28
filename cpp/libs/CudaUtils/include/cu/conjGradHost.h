#ifndef LIBS_CUDAUTILS_INCLUDE_CU_CONJGRADHOST
#define LIBS_CUDAUTILS_INCLUDE_CU_CONJGRADHOST

#include <memory>

#include <linalg/csrMatrix.h>

namespace cu
{
    class csrF;
    class ConjugateGradientF;
    struct CgCudaBuffs;

    class ConjGradHost
    {
        std::unique_ptr<csrF> cudaMat;
        std::unique_ptr<ConjugateGradientF> cudaCg;
        std::unique_ptr<CgCudaBuffs> buffs;

    public:
        ConjGradHost(const linalg::CsrMatrix<float> & m);

        ~ConjGradHost();

        float solve(const std::vector<float> & rhs, std::vector<float> & sol,
                    const int maxIters, const float target);
    };
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_CONJGRADHOST */
