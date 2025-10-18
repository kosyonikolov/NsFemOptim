#ifndef LIBS_FEM_INCLUDE_FEM_BORDERS
#define LIBS_FEM_INCLUDE_FEM_BORDERS

#include <algorithm>
#include <vector>

#include <fem/dirichletNode.h>
#include <mesh/concreteMesh.h>

namespace fem
{
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
} // namespace fem

#endif /* LIBS_FEM_INCLUDE_FEM_BORDERS */
