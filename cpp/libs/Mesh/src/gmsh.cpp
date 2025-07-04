#include <mesh/gmsh.h>

#include <algorithm>
#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mesh
{
    template <typename T>
    std::istream & operator>>(std::istream & stream, std::vector<T> & v)
    {
        T val;
        while (stream >> val)
        {
            v.push_back(val);
        }
        stream.clear(); // Reset stream to "good"
        return stream;
    }

    bool startswith(const std::string & str, const std::string & prefix)
    {
        const auto n = str.size();
        const auto m = prefix.size();
        if (m > n)
        {
            return false;
        }
        for (size_t i = 0; i < m; i++)
        {
            if (str[i] != prefix[i])
            {
                return false;
            }
        }
        return true;
    }

    SectionedText sectionText(std::istream & stream)
    {
        SectionedText result;
        result.names.push_back("");

        int numSections = 1;
        std::map<std::string, int> name2Id;
        name2Id[""] = 0;
        std::vector<std::ostringstream> protoContent(1);

        auto switchId = [&](const std::string & name) -> int
        {
            auto it = name2Id.find(name);
            if (it != name2Id.end())
            {
                return it->second;
            }
            const int newId = numSections++;
            protoContent.push_back({});
            result.names.push_back(name);
            name2Id[name] = newId;
            return newId;
        };

        std::string line;
        int currId = 0;
        while (std::getline(stream, line))
        {
            if (line.empty())
            {
                continue;
            }
            if (line[0] == '$')
            {
                if (startswith(line, "$End"))
                {
                    currId = 0;
                }
                else
                {
                    currId = switchId(line.substr(1));
                }
                continue;
            }

            auto & stream = protoContent[currId];
            stream << line << "\n";
        }

        result.content.resize(numSections);
        for (int i = 0; i < numSections; i++)
        {
            result.content[i] = protoContent[i].str();
        }
        result.name2id = std::move(name2Id);

        return result;
    }

    bool parseLine(std::istringstream &)
    {
        return true;
    }

    template <typename T, typename... Args>
    bool parseLine(std::istringstream & iss, T & val, Args &&... args)
    {
        if (!(iss >> val))
        {
            return false;
        }
        return parseLine(iss, args...);
    }

    template <typename... Args>
    bool parseLine(const std::string & str, Args &&... args)
    {
        std::istringstream iss(str);
        return parseLine(iss, args...);
    }

    template <typename... Args>
    bool parseLineFrom(std::istringstream & iss, Args &&... args)
    {
        std::string line;
        if (!std::getline(iss, line))
        {
            return false;
        }
        return parseLine(line, args...);
    }

    NodeSection parseNodeSection(const std::string & text)
    {
        std::istringstream iss(text);
        NodeSection result;

        struct
        {
            int numBlocks = 0;
            int numNodes = 0;
            int minId = 0;
            int maxId = 0;
        } header;

        // numEntityBlocks(size_t) numNodes(size_t)  minNodeTag(size_t) maxNodeTag(size_t)
        if (!parseLineFrom(iss, header.numBlocks, header.numNodes, header.minId, header.maxId))
        {
            throw std::invalid_argument("Failed to parse header for Nodes");
        }

        for (int i = 0; i < header.numBlocks; i++)
        {
            // entityDim(int) entityTag(int) parametric(int; 0 or 1) numNodesInBlock(size_t)
            int entityDim = 0;
            int entityTag = 0;
            int parametric = 0;
            int numNodesInBlock = 0;
            if (!parseLineFrom(iss, entityDim, entityTag, parametric, numNodesInBlock))
            {
                std::invalid_argument(std::format("Failed to parse entity info with ID{}", i));
            }

            std::vector<ParsedNode> parsedNodes(numNodesInBlock);
            for (int j = 0; j < numNodesInBlock; j++)
            {
                if (!parseLineFrom(iss, parsedNodes[j].id))
                {
                    std::invalid_argument(
                        std::format("Failed to parse node #{}'s tag from entity info with ID{}", j, i));
                }
            }
            for (int j = 0; j < numNodesInBlock; j++)
            {
                if (!parseLineFrom(iss, parsedNodes[j].x, parsedNodes[j].y, parsedNodes[j].z))
                {
                    std::invalid_argument(
                        std::format("Failed to parse node #{}'s coords from entity info with ID{}", j, i));
                }
            }
            result.nodes.insert(result.nodes.end(), parsedNodes.begin(), parsedNodes.end());
        }

        return result;
    }

    template <typename T>
    void parseVec(std::istringstream & stream, std::vector<T> & v)
    {
        T val;
        while (stream >> val)
        {
            v.push_back(val);
        }
    }

    ElementSection parseElementSection(const std::string & text)
    {
        std::istringstream iss(text);
        ElementSection result;

        struct
        {
            int numBlocks = 0;
            int numElements = 0;
            int minId = 0;
            int maxId = 0;
        } header;

        // numEntityBlocks(size_t) numElements(size_t) minElementTag(size_t) maxElementTag(size_t)
        if (!parseLineFrom(iss, header.numBlocks, header.numElements, header.minId, header.maxId))
        {
            throw std::invalid_argument("Failed to parse header for Nodes");
        }

        for (int i = 0; i < header.numBlocks; i++)
        {
            // entityDim(int) entityTag(int) elementType(int; see below)  numElementsInBlock(size_t)
            int entityDim = 0;
            int entityTag = 0;
            int elementType = 0;
            int numElementsInBlock = 0;
            if (!parseLineFrom(iss, entityDim, entityTag, elementType, numElementsInBlock))
            {
                throw std::invalid_argument("Aaaaa");
            }

            std::vector<ParsedElement> elements(numElementsInBlock);
            for (int j = 0; j < numElementsInBlock; j++)
            {
                // elementTag(size_t) nodeTag(size_t) ...
                auto & e = elements[j];
                e.entity = entityTag;

                std::string line;
                if (!std::getline(iss, line))
                {
                    throw std::invalid_argument("aaaa");
                }
                std::istringstream iss1(line);
                if (!(iss1 >> e.id))
                {
                    throw std::invalid_argument("aaaaaa");
                }
                int val;
                while (iss1 >> val)
                {
                    e.points.push_back(val);
                }
            }

            result.elements.insert(result.elements.end(), elements.begin(), elements.end());
        }

        return result;
    }

    PhysicsSection parsePhysicsSection(const std::string & text)
    {
        std::istringstream iss(text);
        PhysicsSection result;

        int numNames;
        if (!parseLineFrom(iss, numNames))
        {
            throw std::invalid_argument("Failed to parse number of names in PhysicalSection");
        }

        int dim;
        int tag;
        std::string name;
        for (int i = 0; i < numNames; i++)
        {
            if (!parseLineFrom(iss, dim, tag, name))
            {
                throw std::runtime_error("Failed to parse line from physics section");
            }
            if (name.size() < 2)
            {
                throw std::runtime_error("Name too short"); // quotes at both ends
            }
            result.names.push_back({dim, tag, name.substr(1, name.length() - 2)});
        }

        return result;
    }

    EntitySection parseEntitySection(const std::string & text)
    {
        std::istringstream iss(text);
        EntitySection result;

        int numPoints, numCurves, numSurfaces, numVolumes;
        if (!parseLineFrom(iss, numPoints, numCurves, numSurfaces, numVolumes))
        {
            throw std::runtime_error("Failed to parse entities header");
        }

        result.points.resize(numPoints);
        result.curves.resize(numCurves);
        result.surfaces.resize(numSurfaces);
        result.volumes.resize(numVolumes);

        auto & points = result.points;
        for (int i = 0; i < numPoints; i++)
        {
            size_t numTags;
            auto & p = points[i];
            if (!parseLineFrom(iss, p.tag, p.x, p.y, p.z, numTags, p.physicalTags))
            {
                throw std::runtime_error("Failed to parse point");
            }
            if (numTags != p.physicalTags.size())
            {
                throw std::runtime_error("Point's physical tags have an incorrect size");
            }
        }

        auto parseHighDimensional = [&](int & tag, double & minX, double & minY, double & minZ, double & maxX,
                                        double & maxY, double & maxZ, std::vector<int> & physicalTags,
                                        std::vector<int> & boundingTags) -> bool
        {
            std::string line;
            if (!std::getline(iss, line))
            {
                return false;
            }

            std::istringstream lineSs(line);
            if (!(lineSs >> tag >> minX >> minY >> minZ >> maxX >> maxY >> maxZ))
            {
                return false;
            }

            // Physical tags
            int numTags;
            if (!(lineSs >> numTags))
            {
                return false;
            }
            int val;
            for (int i = 0; i < numTags; i++)
            {
                if (!(lineSs >> val))
                {
                    return false;
                }
                physicalTags.push_back(val);
            }

            // Bounding tags
            if (!(lineSs >> numTags))
            {
                return false;
            }
            for (int i = 0; i < numTags; i++)
            {
                if (!(lineSs >> val))
                {
                    return false;
                }
                boundingTags.push_back(val);
            }

            return true;
        };

        auto & curves = result.curves;
        for (int i = 0; i < numCurves; i++)
        {
            auto & c = curves[i];
            if (!parseHighDimensional(c.tag, c.minX, c.minY, c.minZ, c.maxX, c.maxY, c.maxZ, c.physicalTags,
                                      c.boundingPoints))
            {
                throw std::runtime_error("Failed to parse curve");
            }
        }

        auto & surfaces = result.surfaces;
        for (int i = 0; i < numSurfaces; i++)
        {
            auto & s = surfaces[i];
            if (!parseHighDimensional(s.tag, s.minX, s.minY, s.minZ, s.maxX, s.maxY, s.maxZ, s.physicalTags,
                                      s.boundingCurves))
            {
                throw std::runtime_error("Failed to parse surface");
            }
        }

        auto & volumes = result.volumes;
        for (int i = 0; i < numVolumes; i++)
        {
            auto & v = volumes[i];
            if (!parseHighDimensional(v.tag, v.minX, v.minY, v.minZ, v.maxX, v.maxY, v.maxZ, v.physicalTags,
                                      v.boundingSurfaces))
            {
                throw std::runtime_error("Failed to parse volume");
            }
        }

        return result;
    }

    void dumpPointsAndElements2d(const std::string & prefix, std::vector<ParsedNode> & nodes,
                                 std::vector<ParsedElement> & elements)
    {
        std::sort(nodes.begin(), nodes.end(), [](const auto & a, const auto & b) { return a.id < b.id; });
        std::sort(elements.begin(), elements.end(), [](const auto & a, const auto & b) { return a.id < b.id; });

        std::ofstream nodeFile(prefix + "nodes.txt");
        for (const auto & node : nodes)
        {
            nodeFile << std::format("{}\t{}\t{}\n", node.id, node.x, node.y);
        }

        std::ofstream borderElements(prefix + "borders.txt");
        std::ofstream elementsFile(prefix + "elements.txt");
        for (const auto & e : elements)
        {
            const auto n = e.points.size();
            if (n == 2)
            {
                borderElements << e.entity << "\t" << e.id;
                for (auto p : e.points)
                {
                    borderElements << "\t" << p;
                }
                borderElements << "\n";
            }
            else if (n == 3)
            {
                elementsFile << e.id;
                for (auto p : e.points)
                {
                    elementsFile << "\t" << p;
                }
                elementsFile << "\n";
            }
        }
    };

    void dumpPhysicalGroups(const std::string & fileName, const PhysicsSection & phys)
    {
        std::ofstream file(fileName);
        for (const auto & name : phys.names)
        {
            file << name.id << "\t" << name.name << "\n";
        }
    }

    Gmsh parseGmsh(const std::string & meshFileName)
    {
        std::ifstream file(meshFileName);
        if (!file.is_open())
        {
            throw std::runtime_error(std::format("Failed to open mesh file [{}]\n", meshFileName));
        }

        auto sectioned = sectionText(file);

        auto nodesSectionText = sectioned.contentOf("Nodes");
        if (!nodesSectionText)
        {
            throw std::runtime_error("No Nodes section!");
        }

        auto elementSectionText = sectioned.contentOf("Elements");
        if (!elementSectionText)
        {
            throw std::runtime_error("No Elements section!");
        }

        Gmsh result;

        result.nodeSection = parseNodeSection(nodesSectionText.value());
        result.elementSection = parseElementSection(elementSectionText.value());

        auto entitiesText = sectioned.contentOf("Entities");
        if (entitiesText)
        {
            result.entitySection = parseEntitySection(entitiesText.value());
        }
        else
        {
            throw std::runtime_error("No Entities section!");
        }

        auto physText = sectioned.contentOf("PhysicalNames");
        if (physText)
        {
            result.physicsSection = parsePhysicsSection(physText.value());
        }
        else
        {
            throw std::runtime_error("No physical section!");
        }

        return result;
    }
} // namespace mesh