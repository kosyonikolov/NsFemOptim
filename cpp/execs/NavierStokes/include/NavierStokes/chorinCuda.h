#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_CHORINCUDA
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_CHORINCUDA

#include <mesh/concreteMesh.h>

#include <NavierStokes/abstractOutputHandler.h>
#include <NavierStokes/chorinCudaConfig.h>
#include <NavierStokes/dfgCondtions.h>

void solveNsChorinCuda(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                       const DfgConditions & cond, const float timeStep0, const float maxT,
                       const ChorinCudaConfig & cfg, AbstractOutputHandler & outputHandler);

#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_CHORINCUDA */
