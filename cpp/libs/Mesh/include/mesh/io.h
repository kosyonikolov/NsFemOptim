#ifndef INCLUDE_MESH_IO
#define INCLUDE_MESH_IO

#include <mesh/triMesh.h>

namespace mesh
{
    struct Gmsh;

    TriangleMesh parseTriangleGmsh(const Gmsh & gmsh);

    TriangleMesh parseTriangleGmsh(const std::string & fileName);
}

#endif
