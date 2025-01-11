#include <mesh/io.h>

#include <algorithm>
#include <stdexcept>

#include <mesh/gmsh.h>
#include <mesh/triMesh.h>

namespace mesh
{
    // clang-format off
    template <typename T>
    concept IdStruct = requires(T t) 
    {
        { t.id };
    };

    template <typename T>
    concept Entity = requires(T t)
    {
        {t.tag} -> std::convertible_to<int>;
        {t.physicalTags} -> std::convertible_to<std::vector<int>>;
    };
    // clang-format on

    template <typename T>
        requires IdStruct<T>
    std::vector<T> sortById(const std::vector<T> & v)
    {
        auto result = v;
        std::sort(result.begin(), result.end(), [](const T & a, const T & b) { return a.id < b.id; });
        return result;
    }

    template <typename T>
        requires IdStruct<T>
    bool checkContiguousIds(const std::vector<T> & v)
    {
        const size_t n = v.size();
        for (size_t i = 1; i < n; i++)
        {
            if (v[i].id != v[i - 1].id + 1)
            {
                return false;
            }
        }
        return true;
    }

    template<Entity E>
    void updateEntity2GroupMap(std::map<int, int> & m, const std::vector<E> & v, const int offset)
    {
        for (const E & e : v)
        {
            if (e.physicalTags.empty())
            {
                continue;
            }
            const int group = e.physicalTags[0] - offset;
            m[e.tag] = group;
        }
    }

    TriangleMesh parseTriangleGmsh(const Gmsh & gmsh)
    {
        const auto srcNodes = sortById(gmsh.nodeSection.nodes);
        const auto srcElements = sortById(gmsh.elementSection.elements);
        const auto srcGroups = sortById(gmsh.physicsSection.names);

        if (!checkContiguousIds(srcNodes))
        {
            throw std::runtime_error("Node IDs are not contiguous");
        }
        if (!checkContiguousIds(srcElements))
        {
            throw std::runtime_error("Element IDs are not contiguous");
        }
        if (!checkContiguousIds(srcGroups))
        {
            throw std::runtime_error("Physical group IDs are not contiguous");
        }

        const int nodeIdOffset = srcNodes[0].id;
        const int elemIdOffset = srcElements[0].id;
        const int groupIdOffset = srcGroups[0].id;

        const int numNodes = srcNodes.size();
        const int numAllElements = srcElements.size();

        TriangleMesh result;
        result.nodes.resize(numNodes);
        for (int i = 0; i < numNodes; i++)
        {
            result.nodes[i].x = srcNodes[i].x;
            result.nodes[i].y = srcNodes[i].y;
        }

        // Create a map from entity tag to group id - for border elements
        std::map<int, int> entity2group;
        updateEntity2GroupMap(entity2group, gmsh.entitySection.points, groupIdOffset);
        updateEntity2GroupMap(entity2group, gmsh.entitySection.curves, groupIdOffset);
        updateEntity2GroupMap(entity2group, gmsh.entitySection.surfaces, groupIdOffset);
        updateEntity2GroupMap(entity2group, gmsh.entitySection.volumes, groupIdOffset);

        // Extract border elements as a list of edges
        // Do this first so that we can assign each edge to a triangle later
        struct Edge
        {
            int to;
            int group;
        };
        std::vector<std::vector<Edge>> edges(numNodes);
        
        int numBorder = 0;
        for (const auto & elem : srcElements)
        {
            if (elem.points.size() != 2)
            {
                continue;
            }
            numBorder++;
            const int from = elem.points[0] - nodeIdOffset;
            const int to = elem.points[1] - nodeIdOffset;
            const int group = entity2group[elem.entity];
            edges[from].push_back({to, group});
        }

        // Parse internal elements
        int numInternal = 0;
        for (const auto & elem : srcElements)
        {
            if (elem.points.size() != 3)
            {
                continue;
            }
            numInternal++;
            std::array<int, 3> points;
            for (size_t i = 0; i < points.size(); i++)
            {
                points[i] = 
            }
        }

        return result;
    }

    TriangleMesh parseTriangleGmsh(const std::string & fileName)
    {
        auto gmsh = parseGmsh(fileName);
        return parseTriangleGmsh(gmsh);
    }
} // namespace mesh