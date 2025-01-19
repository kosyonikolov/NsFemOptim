#ifndef LIBS_ELEMENT_INCLUDE_ELEMENT_CALC
#define LIBS_ELEMENT_INCLUDE_ELEMENT_CALC

#include <element/type.h>

namespace el
{
    using ShapeFn = void(*)(const float, const float, float *);
    using ShapeGradFn = void(*)(const float, const float, float *, float *);
    using ValueFn = float(*)(const float, const float, const float *);

    template<Type t>
    constexpr int dof();

    template<> constexpr int dof<Type::P0>()
    {
        return 1;
    }

    template<> constexpr int dof<Type::P1>()
    {
        return 3;
    }

    template<> constexpr int dof<Type::P2>()
    {
        return 6;
    }

    int dof(const Type & t);

    // Calculates the shape functions on the reference triangles at point (x, y)
    template<Type t>
    void shape(const float x, const float y, float * dst);

    template<Type t>
    void shapeGrad(const float x, const float y, float * dstX, float * dstY);

    template<Type t>
    float value(const float x, const float y, const float * nodeValues);

    ShapeFn getShapeFunction(const Type t);

    ShapeGradFn getShapeGradFunction(const Type t);

    ValueFn getValueFunction(const Type t);
}

#endif /* LIBS_ELEMENT_INCLUDE_ELEMENT_CALC */
