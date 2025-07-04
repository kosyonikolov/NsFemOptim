#ifndef LIBS_CUDAUTILS_INCLUDE_CU_SOLVERS_ABSTRACTSOLVER
#define LIBS_CUDAUTILS_INCLUDE_CU_SOLVERS_ABSTRACTSOLVER

#include <cu/vec.h>

namespace cu
{
    class AbstractSolver
    {
    public:
        virtual int getNumCh() const = 0;

        // ch == -1 -> get combined MSE
        // everything else -> get for specific channel
        virtual float getLastMse(const int ch = -1) const = 0;

        virtual int getLastIterations() const = 0;

        virtual cu::vec<float> & getRhs() = 0;

        virtual cu::vec<float> & getSol() = 0;

        virtual void solve() = 0;

        virtual void setMaxIters(const int n) = 0;

        virtual void setTargetMse(const float mse) = 0;

        virtual void setMseCheckInterval(const int mseMod) = 0;

        virtual ~AbstractSolver() {};
    };
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_SOLVERS_ABSTRACTSOLVER */
