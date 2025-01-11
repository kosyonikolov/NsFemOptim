#ifndef LIBS_MESH_INCLUDE_MESH_AFFINETRANSFORM
#define LIBS_MESH_INCLUDE_MESH_AFFINETRANSFORM

#include <mesh/point.h>

namespace mesh
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
};

#endif /* LIBS_MESH_INCLUDE_MESH_AFFINETRANSFORM */
