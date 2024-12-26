#ifndef INCLUDE_MESH_IO
#define INCLUDE_MESH_IO

#include <mesh/triMesh.h>

namespace mesh
{
    TriangleMesh parseTriangularGmsh(const std::string & fileName);
}

#endif
