#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_LOG
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_LOG

#include <array>
#include <string>
#include <fstream>

struct LogEntry
{
    int id;

    // Computation time
    float tTotal;

    float tTentative;
    int itersTentative;
    std::array<float, 2> mseTentative;

    float tPressure;
    int itersPressure;
    float msePressure;

    float tFinal;
    int itersFinal;
    std::array<float, 2> mseFinal;
};

class Log
{
    std::ofstream file;

public:
    Log(const std::string & fileName);

    void add(const LogEntry & entry);
};

#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_LOG */
