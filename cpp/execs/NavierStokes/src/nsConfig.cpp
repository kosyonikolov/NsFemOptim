#include <NavierStokes/nsConfig.h>

#include <fstream>
#include <stdexcept>

#include <utils/configParser.h>

NsConfig parseNsConfig(const std::string & fileName)
{
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open config file");
    }

    u::ConfigParser parser;
    parser.populate(file);
    NsConfig result;
#define PARSE(x) parser.parse(#x, result.x)
    PARSE(viscosity);
    PARSE(peakVelocity);
    PARSE(maxT);
    PARSE(tau);

    PARSE(output.frameStep);
    PARSE(output.velocityStep);
    PARSE(output.velocityScale);
    PARSE(output.imgScale);
#undef PARSE

    return result;
}