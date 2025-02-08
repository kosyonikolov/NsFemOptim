#include <utils/configParser.h>

namespace u
{
    std::string trim(const std::string & s)
    {
        const size_t n = s.size();
        if (n < 1)
        {
            return "";
        }

        size_t i0 = 0;
        while (i0 < n && std::isspace(s[i0]))
        {
            i0++;
        }
        if (i0 >= n)
        {
            return "";
        }
        
        int i1 = n - 1;
        while (i1 >= 0 && std::isspace(s[i1]))
        {
            i1--;
        }
        if (i1 < 0)
        {
            return "";
        }
        return s.substr(i0, i1 - i0 + 1);
    }

    bool parseKeyValue(const std::string & s, std::string & key, std::string & value)
    {
        auto sepIdx = s.find('=');
        if (sepIdx == s.npos)
        {
            return false;
        }
        auto secondSepIdx = s.find('=', sepIdx + 1);
        if (secondSepIdx != s.npos)
        {
            return false;
        }
        
        key = trim(s.substr(0, sepIdx));
        value = trim(s.substr(sepIdx + 1));
        return key != "" && value != "";
    } 

    void ConfigParser::populate(std::istream & stream)
    {
        std::string line;
        std::string key, value;
        while (std::getline(stream, line))
        {
            if (line == "" || line[0] == '#')
            {
                continue;
            }

            if (!parseKeyValue(line, key, value))
            {
                continue;
            }

            m[key] = value;
        }
    }

    bool ConfigParser::parse(const std::string & key, std::string & v)
    {
        auto it = m.find(key);
        if (it == m.end())
        {
            return false;
        }
        v = it->second;
        return true;
    }
}