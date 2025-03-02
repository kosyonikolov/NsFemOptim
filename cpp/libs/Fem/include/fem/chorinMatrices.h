#ifndef LIBS_FEM_INCLUDE_FEM_CHORINMATRICES
#define LIBS_FEM_INCLUDE_FEM_CHORINMATRICES

#include <linalg/sparseBuilder.h>
#include <linalg/sparseDokBuilder.h>

#include <mesh/concreteMesh.h>

namespace fem
{
    template <typename F>
    struct ChorinMatrices
    {
        linalg::SparseMatrixBuilder<F> velocityMass;
        linalg::SparseMatrixBuilder<F> velocityStiffness;
        linalg::SparseMatrixBuilder<F> pressureStiffness;
        linalg::SparseMatrixBuilder<F> velocityPressureDiv;
        linalg::SparseMatrixBuilder<F> pressureVelocityDiv;
    };

    template <typename F>
    struct ChorinMatricesDok
    {
        linalg::SparseMatrixDokBuilder<F> velocityMass;
        linalg::SparseMatrixDokBuilder<F> velocityStiffness;
        linalg::SparseMatrixDokBuilder<F> pressureStiffness;
        linalg::SparseMatrixDokBuilder<F> velocityPressureDiv;
        linalg::SparseMatrixDokBuilder<F> pressureVelocityDiv;
    };

    template <typename F>
    ChorinMatrices<F> buildChorinMatrices(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                                          const int integrationDegree);

    template <typename F>
    ChorinMatricesDok<F> buildChorinMatricesDok(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                                                const int integrationDegree);
}; // namespace fem

#endif /* LIBS_FEM_INCLUDE_FEM_CHORINMATRICES */
