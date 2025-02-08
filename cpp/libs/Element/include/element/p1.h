#ifndef LIBS_ELEMENT_INCLUDE_ELEMENT_P1
#define LIBS_ELEMENT_INCLUDE_ELEMENT_P1

#include <element/element.h>

namespace el
{
    class P1 : public Element
    {
        std::vector<Point> allNodes;

    public:
        P1();

        Type type() const;
        
        int ptsPerSide() const;

        std::span<const Point> internalNodes() const;

        std::span<const Point> nodes() const;

        int dof() const;

        void shape(const float x, const float y, float * dst) const;

        void grad(const float x, const float y, float * dstX, float * dstY) const;

        float value(const float x, const float y, const float * nodeValues) const;
    };
}

#endif /* LIBS_ELEMENT_INCLUDE_ELEMENT_P0 */