#include <element/element.h>

#include <cassert>

namespace el
{
    std::vector<Point> createTriangleBorderNodes(const int ptsPerSide)
    {
        assert(ptsPerSide != 1); // Nonconforming P1 not supported yet
        if (ptsPerSide < 2)
        {
            return {};
        }

        const int extraNodesPerSide = ptsPerSide - 2;
        const int borderNodes = 3 + 3 * extraNodesPerSide;

        std::vector<Point> result(borderNodes);
        
        const int sideStep = ptsPerSide - 1;
        // Corners
        result[0 * sideStep] = {0, 0};
        result[1 * sideStep] = {1, 0};
        result[2 * sideStep] = {0, 1};
        
        // Extra side points
        const float h = 1.0f / (extraNodesPerSide + 1);
        for (int corner = 0; corner < 3; corner++)
        {
            const int nextCorner = (corner + 1) % 3;
            const int i0 = corner * sideStep;
            const Point a = result[i0];
            const Point b = result[nextCorner * sideStep];

            for (int k = 1; k <= extraNodesPerSide; k++)
            {
                const float w = k * h;
                const float compW = 1.0f - w;
                result[i0 + k].x = w * a.x + compW * b.x;
                result[i0 + k].y = w * a.y + compW * b.y;
            }
        }

        return result;
    }
} // namespace mesh