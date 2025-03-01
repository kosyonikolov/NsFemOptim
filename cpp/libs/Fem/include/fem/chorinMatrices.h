#ifndef LIBS_FEM_INCLUDE_FEM_CHORINMATRICES
#define LIBS_FEM_INCLUDE_FEM_CHORINMATRICES

#include <linalg/sparseBuilder.h>

#include <mesh/concreteMesh.h>

namespace fem
{
    template<typename F>
    struct ChorinMatrices
    {
        linalg::SparseMatrixBuilder<F> velocityMass;
        linalg::SparseMatrixBuilder<F> velocityStiffness;
        linalg::SparseMatrixBuilder<F> pressureStiffness;
        linalg::SparseMatrixBuilder<F> velocityPressureDiv;
        linalg::SparseMatrixBuilder<F> pressureVelocityDiv;
    };

    template<typename F>
    ChorinMatrices<F> buildChorinMatrices(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                                          const int integrationDegree);
};  

#endif /* LIBS_FEM_INCLUDE_FEM_CHORINMATRICES */
