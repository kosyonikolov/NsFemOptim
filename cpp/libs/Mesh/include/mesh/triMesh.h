#ifndef INCLUDE_MESH_TRI_MESH
#define INCLUDE_MESH_TRI_MESH

#include <vector>
#include <array>
#include <string>

#include <element/point.h>

namespace mesh
{
    struct TriangleSide
    {
        int from;
        int to;
    };

    // clang-format off
    constexpr std::array<TriangleSide, 6> triangleSides = 
    {
        TriangleSide{0, 1},
        TriangleSide{1, 0},
        TriangleSide{1, 2},
        TriangleSide{2, 1},
        TriangleSide{2, 0},
        TriangleSide{0, 2},
    };
    // clang-format on

    struct BorderElement
    {
        // ID of triangle to which this element belongs
        int element;

        // Which side of the triangle does this element refer to
        // Index in triangleSides
        int side;

        // Group ID of element
        int group;
    };

    // Simple triangles - 3 points at the vertices
    // This is a kind of "prototype" mesh that is element agnostic
    // Meshes for concrete elements can be generated from it
    struct TriangleMesh
    {
        std::vector<el::Point> nodes;
        std::vector<std::array<int, 3>> elements;
        std::vector<std::string> groups;
        std::vector<BorderElement> borderElements;
    };
}

#endif
