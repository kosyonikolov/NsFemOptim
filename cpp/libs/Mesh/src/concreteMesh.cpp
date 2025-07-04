#include <mesh/concreteMesh.h>

#include <algorithm>
#include <cassert>

#include <element/affineTransform.h>

namespace mesh
{
    int ConcreteMesh::getElementSize() const
    {
        return baseElement->nodes().size();
    }

    int ConcreteMesh::getBorderElementSize() const
    {
        return baseElement->ptsPerSide();
    }

    void ConcreteMesh::getElement(const int id, int * ids, el::Point * pts) const
    {
        assert(id >= 0 && id < numElements);
        const int elSize = getElementSize();
        const int offset = id * elSize;
        const int * srcIds = elements.data() + offset;

        if (ids)
        {
            std::copy_n(srcIds, elSize, ids);
        }
        if (pts)
        {
            for (int i = 0; i < elSize; i++)
            {
                const int j = srcIds[i];
                pts[i] = nodes[j];
            }
        }
    }

    void ConcreteMesh::getBorderElement(const int id, int & triangleId, int & side, int & group, int * ptsIds, el::Point * pts) const
    {
        assert(id >= 0 && id < numBorderElements);
        const int nPts = baseElement->ptsPerSide();
        const int step = BorderElementOrder::PtsStart + nPts;
        const int offset = id * step;
        const int * src = borderElements.data() + offset;

        triangleId = src[BorderElementOrder::TriangleElement];
        side = src[BorderElementOrder::Side];
        group = src[BorderElementOrder::Group];

        const int * srcIds = src + BorderElementOrder::PtsStart;
        if (ptsIds)
        {
            std::copy_n(srcIds, nPts, ptsIds);
        }
        if (pts)
        {
            for (int i = 0; i < nPts; i++)
            {
                const int j = srcIds[i];
                pts[i] = nodes[j];
            }
        }
    }

    int ConcreteMesh::findGroupId(const std::string & name) const
    {
        auto it = std::find(groups.begin(), groups.end(), name);
        if (it == groups.end())
        {
            return -1;
        }
        return it - groups.begin();
    }

    ConcreteMesh createMesh(const TriangleMesh & triMesh, const el::Element & baseElement)
    {
        ConcreteMesh result;
        result.baseElement = &baseElement;

        // Create all the nodes first, posibly adding new ones

        // First copy the corner nodes
        result.nodes = triMesh.nodes;
        const int nVertices = result.nodes.size();

        // Create all the extra side nodes
        struct Side
        {
            int to;
            std::vector<int> extraPtIds;
        };
        std::vector<std::vector<Side>> graph(nVertices);

        auto addSide = [&](const int ptA, const int ptB)
        {
            assert(ptA != ptB);
            const int from = std::min(ptA, ptB);
            const int to = std::max(ptA, ptB);
            assert(ptA >= 0 && ptA < nVertices);
            assert(ptB >= 0 && ptB < nVertices);

            // Check if the side has already been added
            for (const auto & s : graph[from])
            {
                if (s.to == to)
                {
                    return;
                }
            }

            Side s;
            s.to = to;
            graph[from].push_back(s);
        };

        for (const auto & element : triMesh.elements)
        {
            // Add each side of the element to the graph
            for (int a = 0; a < 3; a++)
            {
                const int b = (a + 1) % 3;
                const int ptA = element[a];
                const int ptB = element[b];
                addSide(ptA, ptB);
            }
        }

        // Loop over each side and create the extra points
        const int extraNodesPerSide = std::max(0, baseElement.ptsPerSide() - 2);
        if (extraNodesPerSide > 0)
        {
            const float h = 1.0f / (extraNodesPerSide + 1);
            for (int idFrom = 0; idFrom < nVertices; idFrom++)
            {
                const el::Point from = result.nodes[idFrom];
                for (auto & s : graph[idFrom])
                {
                    const el::Point to = result.nodes[s.to];
                    for (int k = 1; k <= extraNodesPerSide; k++)
                    {
                        const float w = k * h;
                        const float compW = 1.0f - w;
                        el::Point m;
                        m.x = compW * from.x + w * to.x;
                        m.y = compW * from.y + w * to.y;
                        s.extraPtIds.push_back(result.nodes.size());
                        result.nodes.push_back(m);
                    }
                }
            }
        }

        auto getSideExtraIds = [&](const int idA, const int idB, int * dst, const int count) -> bool
        {
            if (idA < idB)
            {
                for (const Side & s : graph[idA])
                {
                    if (s.to == idB && s.extraPtIds.size() == count)
                    {
                        std::copy_n(s.extraPtIds.begin(), count, dst);
                        return true;
                    }
                }

                // Shouldn't happen
                return false;
            }
            else
            {
                for (const Side & s : graph[idB])
                {
                    if (s.to == idA && s.extraPtIds.size() == count)
                    {
                        for (int i = 0; i < s.extraPtIds.size(); i++)
                        {
                            dst[i] = s.extraPtIds[s.extraPtIds.size() - 1 - i];
                        }
                        return true;
                    }
                }

                // Shouldn't happen
                return false;
            }
        };

        // Create each element, adding internal points if necessary
        result.numElements = triMesh.elements.size();
        const int nodesPerElement = baseElement.nodes().size();
        const int elementBufferSize = nodesPerElement * triMesh.elements.size();
        result.elements.resize(elementBufferSize);
        result.elementTransforms.resize(result.numElements);
        result.invElementTransforms.resize(result.numElements);

        const auto baseInternal = baseElement.internalNodes();
        const int numInternal = baseInternal.size();
        const int elementSideStep = std::max(0, baseElement.ptsPerSide() - 1);

        if (baseElement.type() == el::Type::P0)
        {
            // The original corners do not participate in the mesh
            result.nodes.clear();
        }

        for (int i = 0; i < triMesh.elements.size(); i++)
        {
            const auto & srcIds = triMesh.elements[i];

            // Calc transform
            std::array<el::Point, 3> cornerPts;
            for (int i = 0; i < 3; i++)
            {
                cornerPts[i] = triMesh.nodes[srcIds[i]];
            }
            result.elementTransforms[i] = calcAffineTransformFromRefTriangle(cornerPts.data());
            result.invElementTransforms[i] = el::invertAffineTransform(result.elementTransforms[i]);

            int * ids = result.elements.data() + i * nodesPerElement;
            // Corners
            if (elementSideStep > 0)
            {
                for (int c = 0; c < 3; c++)
                {
                    ids[c * elementSideStep] = srcIds[c];
                }
            }
            
            // Sides
            if (extraNodesPerSide > 0)
            {
                for (int a = 0; a < 3; a++)
                {
                    const int b = (a + 1) % 3;
                    const int idA = srcIds[a];
                    const int idB = srcIds[b];
                    int * extraIds = ids + a * elementSideStep + 1;
                    [[maybe_unused]] const bool ok = getSideExtraIds(idA, idB, extraIds, extraNodesPerSide);
                    assert(ok);
                }
            }

            // Internal - they are exclusive to the each element, so we can create them here
            if (numInternal > 0)
            {
                const auto & transform = result.elementTransforms[i];
                int * internalIds = ids + 3 * elementSideStep;
                for (int k = 0; k < numInternal; k++)
                {
                    el::Point newPoint = transform.apply(baseInternal[k]);
                    internalIds[k] = result.nodes.size();
                    result.nodes.push_back(newPoint);
                }
            }
        }

        // Create border elements
        result.numBorderElements = triMesh.borderElements.size();
        const int borderElementStep = baseElement.ptsPerSide() + 3;
        const int borderBuffSize = borderElementStep * result.numBorderElements;
        result.borderElements.resize(borderBuffSize);
        for (int i = 0; i < result.numBorderElements; i++)
        {
            const auto & src = triMesh.borderElements[i];
            const auto & elemPtIds = triMesh.elements[src.element];
            const auto tSide = triangleSides[src.side];
            const int idA = elemPtIds[tSide.from];
            const int idB = elemPtIds[tSide.to];

            int * dst = result.borderElements.data() + i * borderElementStep;
            dst[ConcreteMesh::BorderElementOrder::TriangleElement] = src.element;
            dst[ConcreteMesh::BorderElementOrder::Side] = src.side;
            dst[ConcreteMesh::BorderElementOrder::Group] = src.group;
            if (borderElementStep > 3)
            {
                dst[ConcreteMesh::BorderElementOrder::PtsStart] = idA;
                dst[borderElementStep - 1] = idB;
                if (extraNodesPerSide > 0)
                {
                    [[maybe_unused]] const bool ok = getSideExtraIds(idA, idB, dst + 4, extraNodesPerSide);
                    assert(ok);
                }
            }
        }

        // Copy group names
        result.groups = triMesh.groups;

        return result;
    }
} // namespace mesh