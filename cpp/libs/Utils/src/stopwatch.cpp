#include <utils/stopwatch.h>

namespace u
{
    std::chrono::steady_clock::time_point Stopwatch::now() const
    {
        return std::chrono::steady_clock::now();
    }

    Stopwatch::Stopwatch()
    {
        last = now();
    }

    void Stopwatch::reset()
    {
        last = now();
    }

    float Stopwatch::millis(const bool reset)
    {
        auto n = now();
        float dt = std::chrono::duration_cast<std::chrono::microseconds>(n - last).count() / 1000.0f;
        if (reset)
        {
            last = n;
        }
        return dt;
    }
}