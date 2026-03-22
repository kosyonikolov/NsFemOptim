#include <OutputInterpolator/config.h>

#include <fstream>
#include <stdexcept>

#include <utils/configParser.h>

Config parseConfig(const std::string & fileName)
{
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open config file");
    }

    u::ConfigParser parser;
    parser.populate(file);
    Config result;
#define PARSE(x) parser.parse(#x, result.x)
    PARSE(ext);
    PARSE(imgScale);
    PARSE(outputWidth);

    PARSE(velocityStep);
    PARSE(velocityScale);

    PARSE(pressureSkipSteps);

    PARSE(numThreads);

    PARSE(dumpPressureStats);
    PARSE(manualPressure);
    PARSE(minPressure);
    PARSE(maxPressure);

    PARSE(sequentialOutput);
#undef PARSE

    return result;
}