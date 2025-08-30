#include <mesh/io.h>

#include <algorithm>
#include <cassert>
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

    class IdMap
    {
        int offset;
        std::vector<int> m;
        std::vector<bool> valid;

    public:
        IdMap(const int minVal, const int maxVal)
        {
            if (minVal > maxVal)
            {
                throw std::invalid_argument("minVal can't be larger than maxVal");
            }
            offset = minVal;
            const int n = maxVal - minVal + 1;
            m.resize(n, -1);
            valid.resize(n, false);
        }

        void add(const int srcId, const int dstId)
        {
            const int idx = srcId - offset;
            if (idx < 0 || idx >= m.size())
            {
                throw std::invalid_argument("Bad src id");
            }
            m[idx] = dstId;
            valid[idx] = true;
        }

        int find(const int srcId) const
        {
            const int idx = srcId - offset;
            if (idx < 0 || idx >= m.size())
            {
                throw std::invalid_argument("Bad src id");
            }
            if (!valid[idx])
            {
                throw std::invalid_argument("No dstId for this srcId");
            }
            return m[idx];
        }
    };

    template <typename T>
        requires IdStruct<T>
    std::vector<T> sortById(const std::vector<T> & v)
    {
        auto result = v;
        std::sort(result.begin(), result.end(), [](const T & a, const T & b)
                  { return a.id < b.id; });
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

    // Make sure that all IDs are contiguous and start from zero
    // The last ID will always be size(v) - 1
    // A map from the old to new IDs is returned
    template <typename T>
        requires IdStruct<T>
    IdMap ensureContiguousIds(std::vector<T> & v)
    {
        if (v.empty())
        {
            IdMap emptyMap(0, 0);
            return emptyMap;
        }

        // Find min and max ID
        int m = v[0].id;
        int M = v[0].id;
        for (const T & elem : v)
        {
            m = std::min<int>(m, elem.id);
            M = std::max<int>(M, elem.id);
        }

        IdMap idMap(m, M);
        const size_t n = v.size();
        for (size_t i = 0; i < n; i++)
        {
            T & curr = v[i];
            const int srcId = v[i].id;
            curr.id = i;
            idMap.add(srcId, i);
        }

        return idMap;
    }

    template <Entity E>
    void updateEntity2GroupMap(std::map<int, int> & m, const std::vector<E> & v, const IdMap & groupIdMap)
    {
        for (const E & e : v)
        {
            if (e.physicalTags.empty())
            {
                continue;
            }
            const int group = groupIdMap.find(e.physicalTags[0]);
            m[e.tag] = group;
        }
    }

    TriangleMesh parseTriangleGmsh(const Gmsh & gmsh)
    {
        if (gmsh.nodeSection.nodes.empty() || 
            gmsh.elementSection.elements.empty() ||
            gmsh.physicsSection.names.empty())
        {
            throw std::invalid_argument("Mesh is missing required sections!");
        }

        auto srcNodes = sortById(gmsh.nodeSection.nodes);
        auto srcElements = sortById(gmsh.elementSection.elements);
        auto srcGroups = sortById(gmsh.physicsSection.names);

        // The IDs are not always contiguous, so we remap them
        IdMap nodeMap = ensureContiguousIds(srcNodes);
        IdMap elementMap = ensureContiguousIds(srcElements);
        IdMap groupMap = ensureContiguousIds(srcGroups);

        assert(srcNodes[0].id == 0);
        assert(srcElements[0].id == 0);
        assert(srcGroups[0].id == 0);

        // Update the elements with the new node IDs
        for (auto & element : srcElements)
        {
            for (auto & nodeId : element.points)
            {
                const int newId = nodeMap.find(nodeId);
                nodeId = newId;
            }
        }

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
            assert(group.id >= 0 && group.id < result.groups.size());
            result.groups[group.id] = group.name;
        }

        // Create a map from entity tag to group id - for border elements
        std::map<int, int> entity2group;
        // !!! Only use the curves - the different groups reuse ID's and we only care about the 1D borders !!!
        updateEntity2GroupMap(entity2group, gmsh.entitySection.curves, groupMap);

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
            const int from = elem.points[0];
            assert(from >= 0 && from < numNodes);
            const int to = elem.points[1];
            assert(to >= 0 && to < numNodes);
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
                points[i] = elem.points[i];
            }
            result.elements.push_back(points);
        }

        const size_t totalParsed = numInternal + numBorder;
        if (totalParsed != numAllElements)
        {
            throw std::runtime_error("Not all elements have been parsed");
        }

        // -1 if no edge exists
        auto getBorderEdge = [&](const int from, const int to) -> Edge *
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