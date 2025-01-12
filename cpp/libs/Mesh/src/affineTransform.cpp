#include <mesh/affineTransform.h>

namespace mesh
{
    AffineTransform calcAffineTransformFromRefTriangle(const Point & p0, const Point & p1, const Point & p2)
    {
        AffineTransform result;

        result.b[0] = p0.x;
        result.b[1] = p0.y;

        result.m[0][0] = p1.x - p0.x;
        result.m[0][1] = p2.x - p0.x;
        result.m[1][0] = p1.y - p0.y;
        result.m[1][1] = p2.y - p0.y;

        return result;
    }

    AffineTransform calcAffineTransformFromRefTriangle(const Point * trianglePoints)
    {
        return calcAffineTransformFromRefTriangle(trianglePoints[0], trianglePoints[1], trianglePoints[2]);
    }
}