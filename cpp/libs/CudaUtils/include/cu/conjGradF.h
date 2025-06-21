#ifndef LIBS_CUDAUTILS_INCLUDE_CU_CONJGRADF
#define LIBS_CUDAUTILS_INCLUDE_CU_CONJGRADF

#include <cu/blas.h>
#include <cu/csrF.h>
#include <cu/sparse.h>
#include <cu/spmv.h>
#include <cu/vec.h>

namespace cu
{
    class ConjugateGradientF
    {
        Sparse sparse;
        Blas blas;

        spmv spmv;
        cu::vec<float> r;

    public:
        ConjugateGradientF(csrF & mat);

        float solve(vec<float> & rhs, vec<float> & x, const int maxIters, const float target);
    };
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_CONJGRADF */
