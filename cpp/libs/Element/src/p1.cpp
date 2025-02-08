#include <element/p1.h>

namespace el
{
    P1::P1()
    {
        allNodes = createTriangleBorderNodes(2);
    }

    Type P1::type() const
    {
        return Type::P1;
    }
        
    int P1::ptsPerSide() const
    {
        return 2;
    }

    std::span<const Point> P1::internalNodes() const
    {
        return {};
    }

    std::span<const Point> P1::nodes() const
    {
        return {allNodes.data(), allNodes.size()};
    }

    int P1::dof() const
    {
        return 3;
    }

    void P1::shape(const float x, const float y, float * dst) const
    {
        dst[0] = 1 - x - y;
        dst[1] = x;
        dst[2] = y;
    }

    void P1::grad(const float, const float, float * dstX, float * dstY) const
    {
        // clang-format off
        //      1 - x - y
        dstX[0] = -1;
        dstY[0] = -1;

        //        x
        dstX[1] = 1;
        dstY[1] = 0;

        //        y
        dstX[2] = 0;
        dstY[2] = 1;
        // clang-format on
    }

    float P1::value(const float x, const float y, const float * nodeValues) const
    {
        constexpr int n = 3;
        std::array<float, n> vals;
        shape(x, y, vals.data());
        float sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += vals[i] * nodeValues[i];
        }
        return sum;
    }
}