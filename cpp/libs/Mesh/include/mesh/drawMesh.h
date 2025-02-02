#ifndef LIBS_MESH_INCLUDE_MESH_DRAWMESH
#define LIBS_MESH_INCLUDE_MESH_DRAWMESH

#include <vector>

#include <opencv2/opencv.hpp>

#include <mesh/concreteMesh.h>

namespace mesh
{
    class AbstractColorScale;
    class Interpolator;
    class TriangleLookup;

    cv::Mat drawMesh(const ConcreteMesh & mesh, const float scale);

    cv::Mat drawValues(const Interpolator & interpolator, const AbstractColorScale & colorScale, const float scale);

    // Color pressure + velocity vector field
    cv::Mat drawCfd(const TriangleLookup & triangleLookup, const std::vector<cv::Scalar> & pressureColors,
                    const float imgScale, const float velocityScale, const float velocityStep,
                    const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                    const std::vector<float> & velocityXy, const std::vector<float> & pressure);
}

#endif /* LIBS_MESH_INCLUDE_MESH_DRAWMESH */
