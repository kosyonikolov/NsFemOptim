#include <mesh/io.h>

#include <algorithm>
#include <stdexcept>
#include <cassert>

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
        // const int elemIdOffset = srcElements[0].id;
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

        // Groups
        result.groups.resize(srcGroups.size());
        for (const auto & group : srcGroups)
        {
            result.groups[group.id - groupIdOffset] = group.name;
        }

        // Create a map from entity tag to group id - for border elements
        std::map<int, int> entity2group;
        // !!! Only use the curves - the different groups reuse ID's and we only care about the 1D borders !!!
        // updateEntity2GroupMap(entity2group, gmsh.entitySection.points, groupIdOffset);
        updateEntity2GroupMap(entity2group, gmsh.entitySection.curves, groupIdOffset);
        // updateEntity2GroupMap(entity2group, gmsh.entitySection.surfaces, groupIdOffset);
        // updateEntity2GroupMap(entity2group, gmsh.entitySection.volumes, groupIdOffset);

        // Extract border elements as an adjacency list
        // Do this first so that we can assign each edge to a triangle later
        struct Edge
        {
            int to;
            int group;
            bool claimed = false;
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
                points[i] = elem.points[i] - nodeIdOffset;
            }
            result.elements.push_back(points);
        }

        const size_t totalParsed = numInternal + numBorder;
        if (totalParsed != numAllElements)
        {
            throw std::runtime_error("Not all elements have been parsed");
        }

        // -1 if no edge exists
        auto getBorderEdge = [&](const int from, const int to) -> Edge*
        {
            assert(from >= 0 && from < numNodes);
            assert(to >= 0 && to < numNodes);

            for (Edge & e : edges[from])
            {
                if (e.to == to)
                {
                    return &e;
                }
            }

            return nullptr;
        };

        // Assign border elements to triangles
        const int nElements = result.elements.size();
        for (int i = 0; i < nElements; i++)
        {
            const auto & pts = result.elements[i];
            for (int side = 0; side < triangleSides.size(); side++)
            {
                const auto & s = triangleSides[side];
                const int from = pts[s.from];
                const int to = pts[s.to];
                Edge * e = getBorderEdge(from, to);
                if (!e)
                {
                    continue;
                }

                // Sanity checks
                assert(e->to == to);
                assert(e->claimed == false);
                
                e->claimed = true;
                BorderElement borderElement;
                borderElement.element = i;
                borderElement.side = side;
                borderElement.group = e->group;
                result.borderElements.push_back(borderElement);
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