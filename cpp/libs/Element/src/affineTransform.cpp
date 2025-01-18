#include <element/affineTransform.h>

#include <stdexcept>

namespace el
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

    AffineTransform invertAffineTransform(const AffineTransform & t)
    {
        const auto & m = t.m;
        const double det = static_cast<double>(m[0][0]) * m[1][1] - static_cast<double>(m[0][1]) * m[1][0];
        if (det == 0)
        {
            throw std::invalid_argument("invertAffineTransform: Determinant is 0");
        }
        const double invDet = 1.0 / det;

        AffineTransform result;
        auto & rm = result.m;
        const double a = invDet * m[1][1];
        const double b = invDet * -m[0][1];
        const double c = invDet * -m[1][0];
        const double d = invDet * m[0][0];
        rm[0][0] = a;
        rm[0][1] = b;
        rm[1][0] = c;
        rm[1][1] = d;

        result.b[0] = -(a * t.b[0] + b * t.b[1]);
        result.b[1] = -(c * t.b[0] + d * t.b[1]);

        return result;
    }
}