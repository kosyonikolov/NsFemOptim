#ifndef LIBS_MESH_INCLUDE_MESH_INTERPOLATOR
#define LIBS_MESH_INCLUDE_MESH_INTERPOLATOR

#include <optional>
#include <vector>

#include <element/calc.h>

#include <mesh/triangleLookup.h>

namespace mesh
{
    class Interpolator
    {
        TriangleLookup lookup;
        
        std::vector<float> values;

        el::ValueFn valueFn;

        mutable int lastElementId = -1;

        // Allocate-once buffers for interpolation
        mutable std::vector<int> ptIds;
        mutable std::vector<float> ptValues;

    public:
        struct InterpolatorRange
        {
            float minX;
            float minY;
            float width;
            float height;
        };

        Interpolator(const ConcreteMesh & mesh, const float h);

        std::optional<float> interpolate(const float x, const float y) const;

        void setValues(const std::vector<float> & values);

        InterpolatorRange getRange() const;
    };
} // namespace mesh

#endif /* LIBS_MESH_INCLUDE_MESH_INTERPOLATOR */
