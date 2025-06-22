#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_BORDERS
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_BORDERS

#include <vector>
#include <algorithm>

#include <mesh/concreteMesh.h>
#include <fem/dirichletNode.h>

float calcDirichletVx(const mesh::ConcreteMesh & mesh, const int nodeId, const int borderId);

template <typename BorderValueFn>
std::vector<fem::DirichletNode> extractDirichletNodes(const mesh::ConcreteMesh & mesh,
                                                      const std::vector<int> borderIds,
                                                      BorderValueFn borderValueFn)
{
    std::vector<fem::DirichletNode> result;

    const int elSize = mesh.getBorderElementSize();
    const int numBorderElems = mesh.numBorderElements;
    std::vector<int> ptIds(elSize);
    std::vector<bool> seen(mesh.nodes.size(), false);
    for (int i = 0; i < numBorderElems; i++)
    {
        int triangle, side, group;
        mesh.getBorderElement(i, triangle, side, group, ptIds.data(), 0);
        auto it = std::find(borderIds.begin(), borderIds.end(), group);
        if (it == borderIds.end())
        {
            continue;
        }
        for (int k = 0; k < elSize; k++)
        {
            const int nodeIdx = ptIds[k];
            if (seen[nodeIdx])
            {
                continue;
            }
            const float val = borderValueFn(mesh, nodeIdx, group);
            seen[nodeIdx] = true;
            result.push_back(fem::DirichletNode{ptIds[k], val});
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

std::vector<int> extractInternalNodes(const int numNodes, const std::vector<fem::DirichletNode> & sortedDirichletNodes);

#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_BORDERS */
