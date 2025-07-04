#ifndef LIBS_CUDAUTILS_INCLUDE_CU_SOLVERS_SOLVERFACTORY
#define LIBS_CUDAUTILS_INCLUDE_CU_SOLVERS_SOLVERFACTORY

#include <string>
#include <memory>

#include <linalg/csrMatrix.h>

#include <cu/solvers/abstractSolver.h>

namespace cu
{
    std::unique_ptr<AbstractSolver> createSolver(const std::string & name, const int numCh, const linalg::CsrMatrix<float> & m,
                                                 const int maxIters, const float targetMse, const int mseCheckInterval);
}

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_SOLVERS_SOLVERFACTORY */
