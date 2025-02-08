#ifndef LIBS_UTILS_INCLUDE_UTILS_CONFIGPARSER
#define LIBS_UTILS_INCLUDE_UTILS_CONFIGPARSER

#include <istream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>

namespace u
{
    class ConfigParser
    {
        std::map<std::string, std::string> m;

    public:
        void populate(std::istream & stream);

        bool parse(const std::string & key, std::string & v);

        template <typename Number>
            requires std::is_integral_v<Number> || std::is_floating_point_v<Number>
        bool parse(const std::string & key, Number & v)
        {
            std::string s;
            if (!parse(key, s))
            {
                return false;
            }
            std::istringstream iss(s);
            Number tmp;
            if (iss >> tmp)
            {
                v = tmp;
                return true;
            }
            return false;
        }
    };
} // namespace u

#endif /* LIBS_UTILS_INCLUDE_UTILS_CONFIGPARSER */
