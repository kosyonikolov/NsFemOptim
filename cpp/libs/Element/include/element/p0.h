#ifndef LIBS_ELEMENT_INCLUDE_ELEMENT_P0
#define LIBS_ELEMENT_INCLUDE_ELEMENT_P0

#include <element/element.h>

namespace el
{
    class P0 : public Element
    {
        const Point internalPt{1.0 / 3, 1.0 / 3};
        
    public:
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
