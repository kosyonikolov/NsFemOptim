#ifndef LIBS_UTILS_INCLUDE_UTILS_STOPWATCH
#define LIBS_UTILS_INCLUDE_UTILS_STOPWATCH

#include <chrono>

namespace u
{
    class Stopwatch
    {
        std::chrono::steady_clock::time_point last;

        std::chrono::steady_clock::time_point now() const;

    public:
        Stopwatch();

        void reset();

        float millis(const bool reset = false);
    };
}

#endif /* LIBS_UTILS_INCLUDE_UTILS_STOPWATCH */
