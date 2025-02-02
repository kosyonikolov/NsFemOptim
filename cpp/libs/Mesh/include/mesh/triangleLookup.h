#ifndef LIBS_MESH_INCLUDE_MESH_TRIANGLELOOKUP
#define LIBS_MESH_INCLUDE_MESH_TRIANGLELOOKUP

#include <optional>
#include <vector>

#include <mesh/concreteMesh.h>

namespace mesh
{
    class TriangleLookup
    {
    public:
        struct Result
        {
            int triangleId;
            float localX;
            float localY;
        };

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

        void selfCheckSegments();

        int getSegmentId(const float x, const float y) const;

        TriangleLookup(const ConcreteMesh & mesh, const float h);

        std::optional<Result> testTriangle(const int triangleId, const float x, const float y) const;

        std::optional<Result> lookup(const float x, const float y, int * lastTriangleId = 0) const;
    };
}   

#endif /* LIBS_MESH_INCLUDE_MESH_TRIANGLELOOKUP */
