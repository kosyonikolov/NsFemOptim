#ifndef LIBS_MESH_INCLUDE_MESH_INTERPOLATOR
#define LIBS_MESH_INCLUDE_MESH_INTERPOLATOR

#include <vector>
#include <optional>

#include <element/calc.h>

#include <mesh/concreteMesh.h>

namespace mesh
{
    class Interpolator
    {
        // Global coordinates
        float minX, minY;
        float width, height;

        // Segments
        int cols;
        int rows;
        float segmWidth;
        float segmHeight;

        // Element that intersect a segment
        std::vector<std::vector<int>> segmentElements;

        ConcreteMesh mesh;
        std::vector<float> values;

        el::ValueFn valueFn;

        mutable int lastElementId = -1;

        // Allocate-once buffers for interpolation
        mutable std::vector<int> ptIds; 
        mutable std::vector<float> ptValues;

        int getSegmentId(const float x, const float y) const;

        std::optional<float> interpOnElement(const float x, const float y, const int elemId) const;

    public:
        Interpolator(const ConcreteMesh & mesh, const float h);

        std::optional<float> interpolate(const float x, const float y) const;

        void setValues(const std::vector<float> & values);
    };
}

#endif /* LIBS_MESH_INCLUDE_MESH_INTERPOLATOR */
