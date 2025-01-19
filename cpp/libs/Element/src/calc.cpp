#include <element/calc.h>

#include <array>
#include <stdexcept>
#include <format>

namespace el
{
    int dof(const Type & t)
    {
        if (t == Type::P0)
        {
            return dof<Type::P0>();
        }
        if (t == Type::P1)
        {
            return dof<Type::P1>();
        }
        if (t == Type::P2)
        {
            return dof<Type::P2>();
        }
        throw std::invalid_argument(std::format("{}: Bad element type [{}]", __FUNCTION__, static_cast<int>(t)));
    }

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

    // ==========================================
    // =========== Gradient functions ===========
    // ==========================================

    template<> void shapeGrad<Type::P0>(const float, const float, float * dstX, float * dstY)
    {
        dstX[0] = 0;
        dstY[0] = 0;
    }
    
    template<> void shapeGrad<Type::P1>(const float, const float, float * dstX, float * dstY)
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

    template<> void shapeGrad<Type::P2>(const float x, const float y, float * dstX, float * dstY)
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

    template float value<Type::P0>(const float x, const float y, const float * nodeValues);
    template float value<Type::P1>(const float x, const float y, const float * nodeValues);
    template float value<Type::P2>(const float x, const float y, const float * nodeValues);

    ShapeFn getShapeFunction(const Type t)
    {
        if (t == Type::P0)
        {
            return shape<Type::P0>;
        }
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

    ShapeGradFn getShapeGradFunction(const Type t)
    {
        if (t == Type::P0)
        {
            return shapeGrad<Type::P0>;
        }
        if (t == Type::P1)
        {
            return shapeGrad<Type::P1>;
        }
        if (t == Type::P2)
        {
            return shapeGrad<Type::P2>;
        }
        return 0;
    }

    ValueFn getValueFunction(const Type t)
    {
        if (t == Type::P0)
        {
            return value<Type::P0>;
        }
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