#include <element/point.h>

#include <cmath>

namespace el
{
    Point normalize(const Point & p)
    {
        const float norm = p.x * p.x + p.y * p.y;
        const float scale = 1.0f / std::sqrt(norm);
        return Point{scale * p.x, scale * p.y};
    }

    float distance(const Point & a, const Point & b)
    {
        const float dx = a.x - b.x;
        const float dy = a.y - b.y;
        const float d2 = dx * dx + dy * dy;
        return std::sqrt(d2);
    }
}