#ifndef LIBS_CUDAUTILS_INCLUDE_CU_EVENT
#define LIBS_CUDAUTILS_INCLUDE_CU_EVENT

namespace cu
{
    class Event
    {
        cudaEvent_t event;

    public:
        Event();

        void record();

        void sync();

        float elapsedTimeMs(Event & earlier);

        ~Event();
    };
}

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_EVENT */
