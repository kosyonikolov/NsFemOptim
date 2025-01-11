#ifndef LIBS_MESH_INCLUDE_MESH_ELEMENT
#define LIBS_MESH_INCLUDE_MESH_ELEMENT

#include <array>
#include <vector>

#include <mesh/point.h>

namespace mesh
{
    enum class ElementType
    {
        P0,
        P1,
        P2,
    };

    struct Element
    {
        ElementType type;

        int ptsPerSide;

        // Internal nodes on the reference triangle ((0,0), (1,0), (0,1))
        std::vector<Point> internalNodes;

        int getNodeCount() const;

        // <side 0> <side 1> <side 2> <internal>
        std::vector<Point> getAllNodes() const;
    };

    Element createElement(const ElementType type);
}

#endif /* LIBS_MESH_INCLUDE_MESH_ELEMENT */
