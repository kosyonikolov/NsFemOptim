#ifndef LIBS_ELEMENT_INCLUDE_ELEMENT_POINT
#define LIBS_ELEMENT_INCLUDE_ELEMENT_POINT

namespace el
{
    struct Point
    {
        float x;
        float y;
    };

    Point normalize(const Point & p);

    float distance(const Point & a, const Point & b);
}

#endif /* LIBS_ELEMENT_INCLUDE_ELEMENT_POINT */
