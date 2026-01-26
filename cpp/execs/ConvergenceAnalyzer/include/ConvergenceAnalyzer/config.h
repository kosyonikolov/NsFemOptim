#ifndef CONVERGENCEANALYZER_INCLUDE_CONVERGENCEANALYZER_CONFIG
#define CONVERGENCEANALYZER_INCLUDE_CONVERGENCEANALYZER_CONFIG

#include <string>

struct Config
{
    // Replace input IDs with sequential ones
    bool replaceInputIds = false;
};

Config parseConfig(const std::string & fileName);

#endif /* CONVERGENCEANALYZER_INCLUDE_CONVERGENCEANALYZER_CONFIG */
