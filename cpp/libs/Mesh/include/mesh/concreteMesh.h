#ifndef LIBS_MESH_INCLUDE_MESH_CONCRETEMESH
#define LIBS_MESH_INCLUDE_MESH_CONCRETEMESH

#include <mesh/triMesh.h>
#include <mesh/element.h>
#include <mesh/affineTransform.h>

namespace mesh
{
    struct ConcreteMesh
    {
        enum BorderElementOrder : int
        {
            TriangleElement = 0,
            Side,
            Group,
            PtsStart
        };

        Element baseElement;
        std::vector<Point> nodes;

        int numElements;
        // Size = NumElements * NodesPerElement
        // Order is <element 0> <element 1> ...
        // Node order within each element is the same as in baseElement.getAllNodes()
        std::vector<int> elements;

        // Size = NumElements
        // Transforms that map the reference triangle to each element
        std::vector<AffineTransform> elementTransforms;

        std::vector<std::string> groups;

        int numBorderElements;
        // Size = NumBorderElements * (baseElement.ptsPerSide + 3)
        // Order within element is <triangle element id> <side> <group> <pt 0> <pt 1> ...
        // The element+side information is required because of elements with internal nodes
        // With them, normal gradients depend on points that are not on the side
        std::vector<int> borderElements;

        int getElementSize() const;

        int getBorderElementSize() const;

        // Retrieve an element's ids and/or points - pointers can be null if they are not required
        // They should point to buffers with size at least getElementSize()
        void getElement(const int id, int * ids, Point * pts) const;

        void getBorderElement(const int id, int & triangleId, int & side, int & group, 
                              int * ptsIds, Point * pts) const;
    };

    ConcreteMesh createMesh(const TriangleMesh & triMesh, const Element & baseElement);
}

#endif /* LIBS_MESH_INCLUDE_MESH_CONCRETEMESH */
