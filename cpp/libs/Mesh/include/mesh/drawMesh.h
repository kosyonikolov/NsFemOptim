#ifndef LIBS_MESH_INCLUDE_MESH_DRAWMESH
#define LIBS_MESH_INCLUDE_MESH_DRAWMESH

#include <opencv2/opencv.hpp>

#include <mesh/concreteMesh.h>

namespace mesh
{
    cv::Mat drawMesh(const ConcreteMesh & mesh, const float scale);
}

#endif /* LIBS_MESH_INCLUDE_MESH_DRAWMESH */
