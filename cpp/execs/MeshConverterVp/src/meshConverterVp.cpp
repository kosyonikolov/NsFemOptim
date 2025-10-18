#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <stdexcept>

#include <element/affineTransform.h>
#include <element/factory.h>

#include <mesh/colorScale.h>
#include <mesh/concreteMesh.h>
#include <mesh/drawMesh.h>
#include <mesh/gmsh.h>
#include <mesh/interpolator.h>
#include <mesh/io.h>

// Reorder a P2 mesh in the format used by Vasil Pashov's code
// 4                     2
//
// 5    3         --->   4    3
//
// 0    1    2           0    5    1
void reorderElementsForVp(mesh::ConcreteMesh & mesh)
{
    if (mesh.baseElement->type() != el::Type::P2)
    {
        throw std::invalid_argument("Reorder only works on P2 meshes");
    }
    if (mesh.getElementSize() != 6)
    {
        throw std::invalid_argument("Expected mesh to have element size == 6");
    }

    constexpr int elementSize = 6;
    const std::array<int, elementSize> remap = {0, 2, 4, 3, 5, 1};

    const int numElems = mesh.numElements;
    std::array<int, elementSize> old;
    for (int i = 0; i < numElems; i++)
    {
        int * curr = mesh.elements.data() + i * elementSize;
        std::copy_n(curr, elementSize, old.data());
        for (int j = 0; j < elementSize; j++)
        {
            curr[j] = old[remap[j]];
        }

        // Verify that the reoder is correct
        const float eps = 1e-6f;
        for (int j = 3; j < 6; j++)
        {
            const int localOpposite = j - 3;
            const int localA = (localOpposite + 1) % 3;
            const int localB = (localOpposite + 2) % 3;

            const int ptId = curr[j];
            const int aId = curr[localA];
            const int bId = curr[localB];

            auto pt = mesh.nodes[ptId];
            auto a = mesh.nodes[aId];
            auto b = mesh.nodes[bId];

            const float expectedX = 0.5f * (a.x + b.x);
            const float expectedY = 0.5f * (a.y + b.y);
            const float dx = pt.x - expectedX;
            const float dy = pt.y - expectedY;
            if (std::abs(dx) > eps || std::abs(dy) > eps)
            {
                std::cerr << std::format("Verification FAILED for element {}: dx = {}, dy = {}\n", i, dx, dy);
            }
        }
    }
}

int countPressureNodes(const mesh::ConcreteMesh & vpMesh)
{
    constexpr int elementSize = 6;
    if (vpMesh.getElementSize() != 6)
    {
        throw std::invalid_argument("Mesh must have elements of size 6");
    }

    const int numNodes = vpMesh.nodes.size();
    std::vector<bool> visited(numNodes, false);

    const int numElem = vpMesh.numElements;
    std::array<int, elementSize> ids;
    for (int i = 0; i < numElem; i++)
    {
        vpMesh.getElement(i, ids.data(), 0);
        
        // Corner points of the triangles
        for (int j = 0; j < 3; j++)
        {
            const int k = ids[j];
            visited[k] = true;
        }
    }

    const int count = std::count(visited.begin(), visited.end(), true);
    return count;
}

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./MeshConverterVp <msh file>";
    if (argc != 2)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string meshFileName = argv[1];

    auto gmsh = mesh::parseGmsh(meshFileName);
    auto triMesh = mesh::parseTriangleGmsh(gmsh);

    const auto velocityElement = el::createElement(el::Type::P2);
    auto velocityMesh = mesh::createMesh(triMesh, *velocityElement);
    reorderElementsForVp(velocityMesh);

    const int numVelocityNodes = velocityMesh.nodes.size();
    const int numPressureNodes = countPressureNodes(velocityMesh);
    std::cout << std::format("Velocity nodes = {}\nPressure nodes = {}\n", numVelocityNodes, numPressureNodes);

    return 0;
}
