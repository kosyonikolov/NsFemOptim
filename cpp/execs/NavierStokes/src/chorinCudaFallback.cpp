#include <NavierStokes/chorinCuda.h>

#include <stdexcept>

void solveNsChorinCuda(const mesh::ConcreteMesh &, const mesh::ConcreteMesh &,
                       const DfgConditions &, const float, const float,
                       const ChorinCudaConfig &, AbstractOutputHandler &)
{
    throw std::runtime_error("ChorinCUDA method not available - executable build without CUDA!");
}