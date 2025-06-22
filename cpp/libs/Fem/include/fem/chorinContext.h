#ifndef LIBS_FEM_INCLUDE_FEM_CHORINCONTEXT
#define LIBS_FEM_INCLUDE_FEM_CHORINCONTEXT

#include <vector>

#include <linalg/csrMatrix.h>

#include <fem/dirichletNode.h>

namespace fem
{
    struct ChorinContextF
    {
        int numVelocityNodes;
        std::vector<DirichletNode> dirichletVx, dirichletVy;
        std::vector<int> internalVelocityNodes;

        int numPressureNodes;
        std::vector<DirichletNode> dirichletPressure;
        std::vector<int> internalPressureNodes;

        linalg::CsrMatrix<float> velocityMass;
        linalg::CsrMatrix<float> velocityStiffness;
        linalg::CsrMatrix<float> pressureStiffness;
        linalg::CsrMatrix<float> pressureStiffnessInternal;
        linalg::CsrMatrix<float> velocityPressureDiv;
        linalg::CsrMatrix<float> pressureVelocityDiv;
        linalg::CsrMatrix<float> fastConvectionIntegration;
    };
}

#endif /* LIBS_FEM_INCLUDE_FEM_CHORINCONTEXT */
