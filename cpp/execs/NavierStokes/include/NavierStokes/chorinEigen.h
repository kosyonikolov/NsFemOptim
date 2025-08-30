#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_CHORINEIGEN
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_CHORINEIGEN

#include <mesh/concreteMesh.h>

#include <NavierStokes/abstractOutputHandler.h>
#include <NavierStokes/dfgCondtions.h>

void solveNsChorinEigen(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                        const DfgConditions & cond, const float timeStep0, const float maxT,
                        AbstractOutputHandler & outputHandler);

#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_CHORINEIGEN */
