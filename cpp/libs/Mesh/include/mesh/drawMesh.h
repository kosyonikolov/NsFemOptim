#ifndef LIBS_MESH_INCLUDE_MESH_DRAWMESH
#define LIBS_MESH_INCLUDE_MESH_DRAWMESH

#include <opencv2/opencv.hpp>

#include <mesh/concreteMesh.h>

namespace mesh
{
    class AbstractColorScale;
    class Interpolator;

    cv::Mat drawMesh(const ConcreteMesh & mesh, const float scale);

    cv::Mat drawValues(const Interpolator & interpolator, const AbstractColorScale & colorScale, const float scale);
}

#endif /* LIBS_MESH_INCLUDE_MESH_DRAWMESH */
