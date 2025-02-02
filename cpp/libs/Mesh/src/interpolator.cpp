#include <mesh/interpolator.h>

#include <cassert>
#include <cmath>
#include <format>
#include <stdexcept>

namespace mesh
{
    Interpolator::Interpolator(const ConcreteMesh & mesh, const float h) : lookup(mesh, h)
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

        const int elemSize = mesh.getElementSize();
        ptIds.resize(elemSize);
        ptValues.resize(elemSize);
    }

    std::optional<float> Interpolator::interpolate(const float x, const float y) const
    {
        auto t = lookup.lookup(x, y, &lastElementId);
        if (!t)
        {
            return {};
        }

        const int elemSize = lookup.mesh.getElementSize();
        assert(ptIds.size() == elemSize);
        assert(ptValues.size() == elemSize);
        lookup.mesh.getElement(t->triangleId, ptIds.data(), 0);
        for (int i = 0; i < elemSize; i++)
        {
            const int j = ptIds[i];
            ptValues[i] = values[j];
        }

        assert(valueFn);
        const float value = valueFn(t->localX, t->localY, ptValues.data());
        return value;
    }

    void Interpolator::setValues(const std::vector<float> & values)
    {
        const int expectedSize = lookup.mesh.nodes.size();
        if (values.size() != expectedSize)
        {
            throw std::invalid_argument(std::format("{}: Expected values to be of length {}, but got {} instead",
                                                    __FUNCTION__, expectedSize, values.size()));
        }
        this->values = values;
    }

    Interpolator::InterpolatorRange Interpolator::getRange() const
    {
        InterpolatorRange result;
        result.minX = lookup.minX;
        result.width = lookup.width;
        result.minY = lookup.minY;
        result.height = lookup.height;
        return result;
    }
} // namespace mesh