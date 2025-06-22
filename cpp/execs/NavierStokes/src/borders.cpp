#include <NavierStokes/borders.h>

#include <cassert>

float calcDirichletVx(const mesh::ConcreteMesh & mesh, const int nodeId, const int borderId)
{
    assert(borderId >= 0 && borderId < mesh.groups.size());
    const auto & group = mesh.groups[borderId];
    if (group == "Left")
    {
        const auto & node = mesh.nodes[nodeId];
        const float y = node.y;
        const float v = 20 * y * (0.41 - y) / (0.41 * 0.41);
        return v;
    }
    return 0;
}

std::vector<int> extractInternalNodes(const int numNodes, const std::vector<fem::DirichletNode> & sortedDirichletNodes)
{
    assert(std::is_sorted(sortedDirichletNodes.begin(), sortedDirichletNodes.end()));
    std::vector<int> result;

    int i = 0;
    int j = 0;
    while (i < numNodes && j < sortedDirichletNodes.size())
    {
        const int d = sortedDirichletNodes[j].id;
        if (i < d)
        {
            result.push_back(i);
            i++;
        }
        else if (i == d)
        {
            i++;
            j++;
        }
        else if (d < i)
        {
            j++;
        }
    }

    while (i < numNodes)
    {
        result.push_back(i);
        i++;
    }

    return result;
}