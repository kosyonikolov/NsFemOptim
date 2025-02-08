#ifndef LIBS_ELEMENT_INCLUDE_ELEMENT_ELEMENT
#define LIBS_ELEMENT_INCLUDE_ELEMENT_ELEMENT

#include <vector>
#include <span>

#include <element/point.h>
#include <element/type.h>

namespace el
{
    // Interface for triangular elements
    class Element
    {
    public:
        virtual Type type() const = 0;
        
        virtual int ptsPerSide() const = 0;

        // Internal nodes on the reference triangle ((0,0), (1,0), (0,1))
        virtual std::span<const Point> internalNodes() const = 0;

        virtual std::span<const Point> nodes() const = 0;

        virtual int dof() const = 0;

        // Calculate stuff on the reference triangle
        // All output vectors must have size DOF

        virtual void shape(const float x, const float y, float * dst) const = 0;

        virtual void grad(const float x, const float y, float * dstX, float * dstY) const = 0;

        virtual float value(const float x, const float y, const float * nodeValues) const = 0;

        virtual ~Element(){};
    };

    std::vector<Point> createTriangleBorderNodes(const int ptsPerSide);
}

#endif /* LIBS_ELEMENT_INCLUDE_ELEMENT_ELEMENT */
