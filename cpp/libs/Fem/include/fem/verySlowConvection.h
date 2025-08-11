#ifndef LIBS_FEM_INCLUDE_FEM_VERYSLOWCONVECTION
#define LIBS_FEM_INCLUDE_FEM_VERYSLOWCONVECTION

#include <vector>

#include <Eigen/SparseCore>

#include <mesh/concreteMesh.h>
#include <element/triangleIntegrator.h>

namespace fem
{
    // Classical algorithm for assembling the convection matrix
    // The matrix is generated on each call (the most naive approach)
    struct VerySlowConvection
    {
        const mesh::ConcreteMesh & velocityMesh;
        el::TriangleIntegrator & integrator;

        VerySlowConvection(const mesh::ConcreteMesh & velocityMesh, el::TriangleIntegrator & integrator);
        
        Eigen::SparseMatrix<float, Eigen::RowMajor>  calculate(const std::vector<float> & velocityXy);
    };
} // namespace fem

#endif /* LIBS_FEM_INCLUDE_FEM_VERYSLOWCONVECTION */
