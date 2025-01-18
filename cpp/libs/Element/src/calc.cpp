#include <element/calc.h>

#include <array>

namespace el
{
    // ==========================================
    // ============ Shape functions =============
    // ==========================================

    template<> void shape<Type::P0>(const float, const float, float * dst)
    {
        dst[0] = 1;
    }

    template<> void shape<Type::P1>(const float x, const float y, float * dst)
    {
        dst[0] = 1 - x - y;
        dst[1] = x;
        dst[2] = y;
    }

    template<> void shape<Type::P2>(const float x, const float y, float * dst)
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

    template<Type t>
    float value(const float x, const float y, const float * nodeValues)
    {
        static constexpr int n = dof<t>();
        std::array<float, n> vals;
        shape<t>(x, y, vals.data());
        float sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += vals[i] * nodeValues[i];
        }
        return sum;
    }

    template float value<Type::P1>(const float x, const float y, const float * nodeValues);
    template float value<Type::P2>(const float x, const float y, const float * nodeValues);

    ShapeFn getShapeFunction(const Type t)
    {
        if (t == Type::P1)
        {
            return shape<Type::P1>;
        }
        if (t == Type::P2)
        {
            return shape<Type::P2>;
        }
        return 0;
    }

    ValueFn getValueFunction(const Type t)
    {
        if (t == Type::P1)
        {
            return value<Type::P1>;
        }
        if (t == Type::P2)
        {
            return value<Type::P2>;
        }
        return 0;
    }
}