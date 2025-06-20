#ifndef LIBS_LINALG_INCLUDE_LINALG_VECTORS
#define LIBS_LINALG_INCLUDE_LINALG_VECTORS

#include <cmath>

#include <utils/concepts.h>

namespace linalg
{
    template<typename F, u::VectorLike<F> V>
    double normL2(const V & vec, const bool normalize = false)
    {
        double sum = 0;
        const size_t n = vec.size();
        for (size_t i = 0; i < n; i++)
        {
            sum += vec[i] * vec[i];
        }
        sum = std::max(sum, 0.0);
        return normalize ? std::sqrt(sum / n) : std::sqrt(sum);
    }

    template<typename V>
    double dot(const V & a, const V & b)
    {
        const size_t n = a.size();
        assert(n == b.size());
        double sum = 0;
        for (size_t i = 0; i < n; i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }
}

#endif /* LIBS_LINALG_INCLUDE_LINALG_VECTORS */
