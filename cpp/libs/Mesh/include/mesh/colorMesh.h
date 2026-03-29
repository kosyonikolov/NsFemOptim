#ifndef LIBS_MESH_INCLUDE_MESH_COLORMESH
#define LIBS_MESH_INCLUDE_MESH_COLORMESH

#include <vector>

#include <mesh/triMesh.h>

namespace mesh
{
    std::vector<std::vector<int>> colorMeshElements(const TriangleMesh & mesh);
}

#endif /* LIBS_MESH_INCLUDE_MESH_COLORMESH */
