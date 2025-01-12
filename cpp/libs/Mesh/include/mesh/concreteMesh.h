#ifndef LIBS_MESH_INCLUDE_MESH_CONCRETEMESH
#define LIBS_MESH_INCLUDE_MESH_CONCRETEMESH

#include <mesh/triMesh.h>
#include <mesh/element.h>
#include <mesh/affineTransform.h>

namespace mesh
{
    struct ConcreteMesh
    {
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
        // Size = NumBorderElements * (baseElement.ptsPerSide + 1)
        // Order within element is <group> <pt 0> <pt 1> ...
        std::vector<int> borderElements;

        int getElementSize() const;

        int getBorderElementSize() const;

        // Retrieve an element's ids and/or points - pointers can be null if they are not required
        // They should point to buffers with size at least getElementSize()
        void getElement(const int id, int * ids, Point * pts);
    };

    ConcreteMesh createMesh(const TriangleMesh & triMesh, const Element & baseElement);
}

#endif /* LIBS_MESH_INCLUDE_MESH_CONCRETEMESH */
