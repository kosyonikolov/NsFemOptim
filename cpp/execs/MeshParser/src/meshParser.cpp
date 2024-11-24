#include <algorithm>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct SectionedText
{
    std::map<std::string, int> name2id;
    std::vector<std::string> names;
    std::vector<std::string> content;

    std::optional<std::string> contentOf(const std::string & name)
    {
        const auto it = name2id.find(name);
        if (it == name2id.end())
        {
            return {};
        }
        return content[it->second];
    }
};

struct ParsedNode
{
    int id;
    double x, y, z;
};

struct NodeSection
{
    std::vector<ParsedNode> nodes;
};

struct ParsedElement
{
    int id;
    int entity;
    std::vector<int> points;
};

struct ElementSection
{
    std::vector<ParsedElement> elements;
};

struct PhysicalName
{
    int dimension;
    int tag;
    std::string name;
};

struct PhysicsSection
{
    std::vector<PhysicalName> names;
};

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

template <typename T, typename... Args> bool parseLine(std::istringstream & iss, T & val, Args &&... args)
{
    if (!(iss >> val))
    {
        return false;
    }
    return parseLine(iss, args...);
}

template <typename... Args> bool parseLine(const std::string & str, Args &&... args)
{
    std::istringstream iss(str);
    return parseLine(iss, args...);
}

template <typename... Args> bool parseLineFrom(std::istringstream & iss, Args &&... args)
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
                std::invalid_argument(std::format("Failed to parse node #{}'s tag from entity info with ID{}", j, i));
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

template <typename T> void parseVec(std::istringstream & stream, std::vector<T> & v)
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
        result.names.push_back({dim, tag, name});
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

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./MeshParser <msh file>";
    if (argc != 2)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string meshFileName = argv[1];

    std::ifstream file(meshFileName);
    if (!file.is_open())
    {
        std::cerr << std::format("Failed to open mesh file [{}]\n", meshFileName);
        return 1;
    }

    auto sectioned = sectionText(file);

    auto nodesSectionText = sectioned.contentOf("Nodes");
    if (!nodesSectionText)
    {
        std::cerr << "No Nodes section!\n";
        return 1;
    }

    auto elementSectionText = sectioned.contentOf("Elements");
    if (!elementSectionText)
    {
        std::cerr << "No Elements section!\n";
        return 1;
    }

    auto nodeSection = parseNodeSection(nodesSectionText.value());
    auto elementSection = parseElementSection(elementSectionText.value());

    auto physText = sectioned.contentOf("PhysicalNames");
    PhysicsSection physicalSection;
    if (physText)
    {
        physicalSection = parsePhysicsSection(physText.value());
    }
    else
    {
        std::cerr << "No physical section!\n";
    }

    dumpPointsAndElements2d("", nodeSection.nodes, elementSection.elements);

    return 0;
}
