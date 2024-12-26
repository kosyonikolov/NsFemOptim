#ifndef INCLUDE_MESH_TRI_MESH
#define INCLUDE_MESH_TRI_MESH

#include <vector>
#include <array>
#include <string>

namespace mesh
{
    struct Point
    {
        float x;
        float y;
    };

    struct BorderElement
    {
        // ID of triangle to which this element belongs
        int element;

        // Which side of the triangle does this element refer to
        // 0: 0 -> 1
        // 1: 1 -> 0
        // 2: 1 -> 2
        // 3: 2 -> 1
        // 4: 2 -> 0
        // 5: 0 -> 2
        int side;

        // Border ID of element
        int border;
    };

    // Simple triangles - 3 points at the vertices
    struct TriangleMesh
    {
        std::vector<Point> nodes;
        std::vector<std::array<int, 3>> elements;
        std::vector<std::string> borders;
        std::vector<BorderElement> borderElements;
    };
}

#endif
