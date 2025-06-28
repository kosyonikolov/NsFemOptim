#ifndef LIBS_CUDAUTILS_INCLUDE_CU_STOPWATCH
#define LIBS_CUDAUTILS_INCLUDE_CU_STOPWATCH

namespace cu
{
    class Stopwatch
    {
        cudaStream_t stream = 0;
        cudaEvent_t start, stop;

    public:
        Stopwatch(cudaStream_t stream = 0);

        void reset();

        float millis(const bool reset = false);
    };
}

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_STOPWATCH */
