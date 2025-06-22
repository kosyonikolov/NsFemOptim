#ifndef LIBS_FEM_INCLUDE_FEM_FASTCONVECTION
#define LIBS_FEM_INCLUDE_FEM_FASTCONVECTION

#include <vector>

#include <Eigen/SparseCore>

#include <mesh/concreteMesh.h>
#include <element/triangleIntegrator.h>

namespace fem
{
    struct FastConvection
    {
        Eigen::SparseMatrix<float, Eigen::RowMajor> convection;
        Eigen::SparseMatrix<float, Eigen::RowMajor> integration;
        Eigen::Vector<float, Eigen::Dynamic> values;
        Eigen::Vector<float, Eigen::Dynamic> velocity; // x, then y

        FastConvection(const mesh::ConcreteMesh & velocityMesh, el::TriangleIntegrator & integrator);
        
        void update(const std::vector<float> & velocityXy);
    };
} // namespace fem

#endif /* LIBS_FEM_INCLUDE_FEM_FASTCONVECTION */
