#include <mesh/interpolator.h>

#include <cassert>
#include <cmath>
#include <format>
#include <stdexcept>

namespace mesh
{
    int Interpolator::getSegmentId(const float x, const float y) const
    {
        const float x1 = x - minX;
        const float y1 = y - minY;

        if (x1 < 0 || x1 > width || y1 < 0 || y1 > height)
        {
            // Outside of grid
            return -1;
        }

        const float kX = x1 / segmWidth;
        const float kY = y1 / segmHeight;

        const int ix = std::min<int>(kX, cols - 1);
        const int iy = std::min<int>(kY, rows - 1);

        return iy * cols + ix;
    }

    std::optional<float> Interpolator::interpOnElement(const float x, const float y, const int elemId) const
    {
        assert(values.size() == mesh.nodes.size());
        assert(elemId >= 0 && elemId < mesh.numElements);

        const el::Point globalPt(x, y);
        const auto transform = mesh.invElementTransforms[elemId];
        const auto refPt = transform(globalPt);

        // Check if inside triangle
        if (refPt.x < 0 || refPt.x > 1 || refPt.y < 0 || refPt.y > 1 || refPt.x + refPt.y > 1)
        {
            return {};
        }

        const int elemSize = mesh.getElementSize();
        assert(ptIds.size() == elemSize);
        assert(ptValues.size() == elemSize);
        mesh.getElement(elemId, ptIds.data(), 0);
        for (int i = 0; i < elemSize; i++)
        {
            const int j = ptIds[i];
            ptValues[i] = values[j];
        }

        assert(valueFn);
        const float value = valueFn(refPt.x, refPt.y, ptValues.data());
        return value;
    }

    Interpolator::Interpolator(const ConcreteMesh & mesh, const float h) : mesh(mesh)
    {
        if (h <= 0)
        {
            throw std::invalid_argument(std::format("{}: Bad segment size [{}]", __FUNCTION__, h));
        }

        valueFn = el::getValueFunction(mesh.baseElement.type);
        if (!valueFn)
        {
            throw std::invalid_argument(std::format("{}: Couldn't obtain value function (element type = {})",
                                                    __FUNCTION__, static_cast<int>(mesh.baseElement.type)));
        }

        // Create bounding rects for each triangle
        struct Rect
        {
            float minX, minY;
            float maxX, maxY;
        };
        float globalMinX = std::numeric_limits<float>::max();
        float globalMinY = std::numeric_limits<float>::max();
        float globalMaxX = std::numeric_limits<float>::min();
        float globalMaxY = std::numeric_limits<float>::min();
        const int nElems = mesh.numElements;
        std::vector<Rect> boundRects(nElems);
        {
            std::array<el::Point, 3> refPts = {el::Point{0, 0}, el::Point{1, 0}, el::Point{0, 1}};
            for (int i = 0; i < nElems; i++)
            {
                const auto & t = mesh.elementTransforms[i];
                auto & r = boundRects[i];

                auto corner = t(refPts[0]);
                r.minX = r.maxX = corner.x;
                r.minY = r.maxY = corner.y;
                for (int j = 1; j < 3; j++)
                {
                    corner = t(refPts[j]);
                    r.minX = std::min(r.minX, corner.x);
                    r.minY = std::min(r.minY, corner.y);
                    r.maxX = std::max(r.maxX, corner.x);
                    r.maxY = std::max(r.maxY, corner.y);
                }

                globalMinX = std::min(globalMinX, r.minX);
                globalMinY = std::min(globalMinY, r.minY);
                globalMaxX = std::max(globalMaxX, r.maxX);
                globalMaxY = std::max(globalMaxY, r.maxY);
            }
        }

        minX = globalMinX;
        minY = globalMinY;
        width = globalMaxX - globalMinX;
        height = globalMaxY - globalMinY;

        cols = std::ceil(width / h);
        rows = std::ceil(height / h);
        segmWidth = width / cols;
        segmHeight = height / rows;

        const int nSeg = cols * rows;
        segmentElements.resize(nSeg);

        // Match elements with segments
        for (int i = 0; i < nSeg; i++)
        {
            const int ix = i % cols;
            const int iy = i / cols;
            auto & ids = segmentElements[i];

            // Top-left corner of segment
            const float segX = minX + ix * segmWidth;
            const float segY = minY + iy * segmHeight;
            for (int j = 0; j < nElems; j++)
            {
                const auto & r = boundRects[j];
                if (r.maxX < segX || r.maxY < segY)
                {
                    continue;
                }
                if (segX + segmWidth < r.minX || segY + segmHeight < r.minY)
                {
                    continue;
                }
                ids.push_back(j);
            }
        }

        // TODO Perform a self-check of the element here - the points of each element should map to a segment that contains their element

        const int elemSize = mesh.getElementSize();
        ptIds.resize(elemSize);
        ptValues.resize(elemSize);
    }

    std::optional<float> Interpolator::interpolate(const float x, const float y) const
    {
        if (values.empty())
        {
            return {};
        }

        if (x < minX || y < minY || x > minX + width || y > minY + height)
        {
            return {};
        }

        // Try the last element ID
        if (lastElementId != -1)
        {
            assert(lastElementId >= 0 && lastElementId < mesh.numElements);
            const auto test = interpOnElement(x, y, lastElementId);
            if (test)
            {
                return test;
            }
        }

        const int segId = getSegmentId(x, y);
        if (segId < 0)
        {
            return {};
        }
        assert(segId < segmentElements.size());

        // Check each element in the segment
        const auto & elemIds = segmentElements[segId];
        for (const int elemId : elemIds)
        {
            const auto test = interpOnElement(x, y, elemId);
            if (test)
            {
                lastElementId = elemId;
                return test;
            }
        }

        return {};
    }

    void Interpolator::setValues(const std::vector<float> & values)
    {
        const int expectedSize = mesh.nodes.size();
        if (values.size() != expectedSize)
        {
            throw std::invalid_argument(std::format("{}: Expected values to be of length {}, but got {} instead",
                                                    __FUNCTION__, expectedSize, values.size()));
        }
        this->values = values;
    }
} // namespace mesh