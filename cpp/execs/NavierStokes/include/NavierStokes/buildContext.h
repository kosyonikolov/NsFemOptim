#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_BUILDCONTEXT
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_BUILDCONTEXT

#include <fem/chorinContext.h>

#include <mesh/concreteMesh.h>

#include <NavierStokes/dfgCondtions.h>
#include <NavierStokes/solution.h>

fem::ChorinContextF buildChorinContext(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                                       const DfgConditions & cond);

#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_BUILDCONTEXT */
