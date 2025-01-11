#include <mesh/element.h>

#include <stdexcept>
#include <cassert>
#include <algorithm>

namespace mesh
{
    int Element::getNodeCount() const
    {
        assert(ptsPerSide != 1); // Noncomforming not supported yet
        if (ptsPerSide < 2)
        {
            return internalNodes.size();
        }

        const int extraNodesPerSide = ptsPerSide - 2;
        const int borderNodes = 3 + 3 * extraNodesPerSide;
        return borderNodes + internalNodes.size();
    }

    std::vector<Point> Element::getAllNodes() const
    {
        assert(ptsPerSide != 1); // Noncomforming not supported yet
        if (ptsPerSide < 2)
        {
            return internalNodes;
        }

        const int extraNodesPerSide = ptsPerSide - 2;
        const int borderNodes = 3 + 3 * extraNodesPerSide;

        std::vector<Point> result(borderNodes + internalNodes.size());
        
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

        std::copy(internalNodes.begin(), internalNodes.end(), result.begin() + borderNodes);
        return result;
    };

    Element createElement(const ElementType type)
    {
        Element result;
        result.type = type;

        if (type == ElementType::P0)
        {
            result.ptsPerSide = 0;
            result.internalNodes = {Point{1.0 / 3, 1.0 / 3}};
            return result;
        }
        if (type == ElementType::P1)
        {
            result.ptsPerSide = 2;
            return result;
        }
        if (type == ElementType::P2)
        {
            result.ptsPerSide = 3;
            return result;
        }

        throw std::invalid_argument("Invalid element type");
    }
} // namespace mesh