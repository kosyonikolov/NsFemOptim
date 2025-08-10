#ifndef LIBS_FEM_INCLUDE_FEM_SLOWCONVECTION
#define LIBS_FEM_INCLUDE_FEM_SLOWCONVECTION

#include <vector>

#include <Eigen/SparseCore>

#include <mesh/concreteMesh.h>
#include <element/triangleIntegrator.h>

namespace fem
{
    // Classical algorithm for assembling the convection matrix
    // The matrix itself is first created with dummy data
    // Subsequent update calls reuse the sparse structure
    struct SlowConvection
    {
        Eigen::SparseMatrix<float, Eigen::RowMajor> convection;
        const mesh::ConcreteMesh & velocityMesh;
        el::TriangleIntegrator & integrator;

        // DoF x DoF matrices with offsets in the value vector of the convection matrix
        std::vector<cv::Mat> elementIndices;

        SlowConvection(const mesh::ConcreteMesh & velocityMesh, el::TriangleIntegrator & integrator);
        
        void update(const std::vector<float> & velocityXy);
    };
} // namespace fem

#endif /* LIBS_FEM_INCLUDE_FEM_SLOWCONVECTION */
