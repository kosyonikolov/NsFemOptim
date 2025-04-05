#ifndef LIBS_FEM_INCLUDE_FEM_CHORINCSR
#define LIBS_FEM_INCLUDE_FEM_CHORINCSR

#include <linalg/csrMatrix.h>

#include <mesh/concreteMesh.h>

namespace fem
{
    template <typename F>
    struct ChorinCsrMatrices
    {
        linalg::CsrMatrix<F> velocityMass;
        linalg::CsrMatrix<F> velocityStiffness;
        linalg::CsrMatrix<F> pressureStiffness;
        linalg::CsrMatrix<F> velocityPressureDiv;
        linalg::CsrMatrix<F> pressureVelocityDiv;
    };

    template <typename F>
    ChorinCsrMatrices<F> buildChorinCsrMatrices(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                                                const int integrationDegree, const int nThreads);

}; // namespace fem

#endif /* LIBS_FEM_INCLUDE_FEM_CHORINCSR */
