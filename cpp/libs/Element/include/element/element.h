#ifndef LIBS_ELEMENT_INCLUDE_ELEMENT_ELEMENT
#define LIBS_ELEMENT_INCLUDE_ELEMENT_ELEMENT

#include <vector>

#include <element/point.h>
#include <element/type.h>

namespace el
{
    struct Element
    {
        Type type;

        int ptsPerSide;

        // Internal nodes on the reference triangle ((0,0), (1,0), (0,1))
        std::vector<Point> internalNodes;

        int getNodeCount() const;

        // <side 0> <side 1> <side 2> <internal>
        std::vector<Point> getAllNodes() const;
    };

    Element createElement(const Type type);
}

#endif /* LIBS_ELEMENT_INCLUDE_ELEMENT_ELEMENT */
