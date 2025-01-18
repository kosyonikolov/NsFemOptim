#ifndef LIBS_ELEMENT_INCLUDE_ELEMENT_AFFINETRANSFORM
#define LIBS_ELEMENT_INCLUDE_ELEMENT_AFFINETRANSFORM

#include <element/point.h>

namespace el
{
    struct AffineTransform
    {
        float m[2][2];
        float b[2];

        void apply(const Point & src, Point & dst) const
        {
            dst.x = m[0][0] * src.x + m[0][1] * src.y + b[0];
            dst.y = m[1][0] * src.x + m[1][1] * src.y + b[1];
        }

        Point apply(const Point & src) const
        {
            Point res;
            apply(src, res);
            return res;
        }

        Point operator()(const Point & src) const
        {
            return apply(src);
        }
    };

    AffineTransform calcAffineTransformFromRefTriangle(const Point & p0, const Point & p1, const Point & p2);

    AffineTransform calcAffineTransformFromRefTriangle(const Point * trianglePoints);

    AffineTransform invertAffineTransform(const AffineTransform & t);
};

#endif /* LIBS_ELEMENT_INCLUDE_ELEMENT_AFFINETRANSFORM */
