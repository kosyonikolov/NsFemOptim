#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_CHORINCUDA
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_CHORINCUDA

#include <mesh/concreteMesh.h>

#include <NavierStokes/chorinCudaConfig.h>
#include <NavierStokes/dfgCondtions.h>
#include <NavierStokes/solution.h>

Solution solveNsChorinCuda(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                           const DfgConditions & cond, const float timeStep0, const float maxT,
                           const ChorinCudaConfig & cfg);

#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_CHORINCUDA */
