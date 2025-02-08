#include <element/p2.h>

namespace el
{
    P2::P2()
    {
        allNodes = createTriangleBorderNodes(3);
    }

    Type P2::type() const
    {
        return Type::P2;
    }
        
    int P2::ptsPerSide() const
    {
        return 3;
    }

    std::span<const Point> P2::internalNodes() const
    {
        return {};
    }

    std::span<const Point> P2::nodes() const
    {
        return {allNodes.data(), allNodes.size()};
    }

    int P2::dof() const
    {
        return 6;
    }

    void P2::shape(const float x, const float y, float * dst) const
    {
        // term psi0	psi1	psi2	psi3	psi4	psi5
        // 1	1	    0	    0	    0	    0	    0
        // x	-3	    4	    -1	    0	    0	    0
        // y	-3	    0	    0	    0	    -1	    4
        // x2	2	    -4	    2	    0	    0	    0
        // y2	2	    0	    0	    0	    2	    -4
        // xy	4	    -4	    0	    4	    0	    -4

        const float x2 = x * x;
        const float y2 = y * y;
        const float xy = x * y;

        dst[0] = 1 - 3 * x - 3 * y + 2 * x2 + 2 * y2 + 4 * xy;
        dst[1] = 4 * x - 4 * x2 - 4 * xy;
        dst[2] = -x + 2 * x2;
        dst[3] = 4 * xy;
        dst[4] = -y + 2 * y2;
        dst[5] = 4 * y - 4 * y2 - 4 * xy;
    }

    void P2::grad(const float x, const float y, float * dstX, float * dstY) const
    {
        // clang-format off
        //     1 - 3 * x - 3 * y + 2 * x2 + 2 * y2 + 4 * xy
        dstX[0] = -3             + 4 * x           + 4 * y;
        dstY[0] =         -3              + 4 * y  + 4 * x;

        //        4 * x - 4 * x2 - 4 * xy
        dstX[1] = 4     - 8 * x  - 4 * y;
        dstY[1] =                - 4 * x;

        //        -x + 2 * x2
        dstX[2] = -1 + 4 * x;
        dstY[2] = 0;

        //        4 * xy
        dstX[3] = 4 * y;
        dstY[3] = 4 * x;

        //        -y + 2 * y2
        dstX[4] = 0;
        dstY[4] = -1 + 4 * y;

        //        4 * y - 4 * y2 - 4 * xy
        dstX[5] =                - 4 * y;
        dstY[5] = 4     - 8 * y  - 4 * x;
        //clang-format on
    }

    float P2::value(const float x, const float y, const float * nodeValues) const
    {
        constexpr int n = 6;
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