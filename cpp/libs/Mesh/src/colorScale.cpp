#include <mesh/colorScale.h>

#include <stdexcept>

namespace mesh
{
    SimpleColorScale::SimpleColorScale(const float min, const float max, const std::vector<cv::Scalar> & colors)
        : min(min), max(max), colors(colors)
    {
        if (min >= max)
        {
            throw std::invalid_argument("Bad min/max");
        }
        if (colors.size() < 2)
        {
            throw std::invalid_argument("No colors");
        }
    }

    cv::Scalar SimpleColorScale::operator()(const float x) const
    {
        const int n = colors.size();
        const float xC = std::clamp(x, min, max);
        const float idxF = (n - 1) * (xC - min) / (max - min);
        const int idx = std::min<int>(idxF, n - 2);
        const float k = idxF - idx;

        const auto low = colors[idx];
        const auto high = colors[idx + 1];
        
        return (1.0f - k) * low + k * high;
    }

    SimpleColorScale::~SimpleColorScale()
    {

    }
} // namespace mesh