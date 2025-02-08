#include <element/p0.h>

namespace el
{
    Type P0::type() const
    {
        return Type::P0;
    }
        
    int P0::ptsPerSide() const
    {
        return 0;
    }

    std::span<const Point> P0::internalNodes() const
    {
        return {&internalPt, 1};
    }

    std::span<const Point> P0::nodes() const
    {
        return {&internalPt, 1};
    }

    int P0::dof() const
    {
        return 1;
    }

    void P0::shape(const float, const float, float * dst) const
    {
        dst[0] = 1;
    }

    void P0::grad(const float, const float, float * dstX, float * dstY) const
    {
        dstX[0] = 0;
        dstY[0] = 0;
    }

    float P0::value(const float, const float, const float * nodeValues) const
    {
        return nodeValues[0];
    }
}