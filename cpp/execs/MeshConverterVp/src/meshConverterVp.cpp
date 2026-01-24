#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <element/affineTransform.h>
#include <element/factory.h>

#include <fem/borders.h>

#include <mesh/colorScale.h>
#include <mesh/concreteMesh.h>
#include <mesh/drawMesh.h>
#include <mesh/gmsh.h>
#include <mesh/interpolator.h>
#include <mesh/io.h>

struct Borders
{
    int left;
    int right;
    int top;
    int bottom;
    int circle;
};

Borders findBorderIds(const mesh::ConcreteMesh & mesh)
{
    Borders result;

    result.left = mesh.findGroupId("Left");
    if (result.left < 0)
    {
        throw std::invalid_argument("No left border!");
    }

    result.right = mesh.findGroupId("Right");
    if (result.right < 0)
    {
        throw std::invalid_argument("No right border!");
    }

    result.top = mesh.findGroupId("Top");
    if (result.top < 0)
    {
        throw std::invalid_argument("No top border!");
    }

    result.bottom = mesh.findGroupId("Bottom");
    if (result.bottom < 0)
    {
        throw std::invalid_argument("No bottom border!");
    }

    result.circle = mesh.findGroupId("Circle");
    if (result.circle < 0)
    {
        throw std::invalid_argument("No circle border!");
    }

    return result;
}

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

std::vector<int> extractBorderVelocityNodes(const std::vector<fem::DirichletNode> & dirichletNodes)
{
    const int n = dirichletNodes.size();
    std::vector<int> result(n);

    for (int i = 0; i < n; i++)
    {
        result[i] = dirichletNodes[i].id;
    }

    return result;
}

std::vector<int> extractBorderPressureNodes(const mesh::ConcreteMesh & vpMesh, const std::vector<fem::DirichletNode> & dirichletNodes)
{
    // Since the border nodes are extracted from the velocity mesh, we have to discard the extra nodes
    // We assume that the mesh is already reordered in VP format

    constexpr int elementSize = 6;
    if (vpMesh.getElementSize() != elementSize)
    {
        throw std::invalid_argument("Mesh must use an element size of 6");
    }

    const int numNodes = vpMesh.nodes.size();
    std::vector<bool> isPressureNode(numNodes, false);
    const int numElem = vpMesh.numElements;
    for (int i = 0; i < numElem; i++)
    {
        std::array<int, elementSize> ids;
        vpMesh.getElement(i, ids.data(), 0);
        for (int j = 0; j < 3; j++)
        {
            // Corner nodes only - they are present in the pressure mesh
            const int k = ids[j];
            isPressureNode[k] = true;
        }
    }

    std::vector<int> result;
    for (const auto & dn : dirichletNodes)
    {
        if (isPressureNode[dn.id])
        {
            result.push_back(dn.id);
        }
    }

    return result;
}

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./MeshConverterVp <msh file> <peak velocity> [output json filename]";
    if (argc < 3)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string meshFileName = argv[1];
    const std::string peakVelocity = argv[2];
    const std::string outFileName = argc > 3 ? argv[3] : "mesh.json";

    auto gmsh = mesh::parseGmsh(meshFileName);
    auto triMesh = mesh::parseTriangleGmsh(gmsh);

    const auto velocityElement = el::createElement(el::Type::P2);
    auto velocityMesh = mesh::createMesh(triMesh, *velocityElement);

    // Extract border nodes before reordering - don't know if the extract functions will work correctly
    const auto borders = findBorderIds(velocityMesh);
    const std::vector<int> leftBorder = {borders.left};
    const std::vector<int> centralBorders = {borders.top, borders.circle, borders.bottom};
    const std::vector<int> rightBorder = {borders.right};

    auto dirichletZero = [](const mesh::ConcreteMesh &, const int, const int) -> float
    {
        return 0.0f;
    };

    const auto leftBorderNodes = fem::extractDirichletNodes(velocityMesh, leftBorder, dirichletZero);
    const auto centralBorderNodes = fem::extractDirichletNodes(velocityMesh, centralBorders, dirichletZero);
    const auto rightBorderNodes = fem::extractDirichletNodes(velocityMesh, rightBorder, dirichletZero);

    reorderElementsForVp(velocityMesh);

    const int numVelocityNodes = velocityMesh.nodes.size();
    const int numPressureNodes = countPressureNodes(velocityMesh);
    std::cout << std::format("Velocity nodes = {}\nPressure nodes = {}\n", numVelocityNodes, numPressureNodes);

    const auto leftNodes = extractBorderVelocityNodes(leftBorderNodes);
    const auto centralNodes = extractBorderVelocityNodes(centralBorderNodes);
    const auto rightNodes = extractBorderPressureNodes(velocityMesh, rightBorderNodes);

    // Write JSON
    std::ofstream json(outFileName);
    if (!json.is_open())
    {
        std::cerr << "Failed to open output file " << outFileName << "\n";
        return 1;
    }

    // Header 
    json << "{\n\t\"elements\":[\n";

    // Elements
    const int numElem = velocityMesh.numElements;
    std::array<int, 6> ids;
    for (int i = 0; i < numElem; i++)
    {
        velocityMesh.getElement(i, ids.data(), 0);
        json << "\t\t[\n";
        for (int k = 0; k < ids.size(); k++)
        {
            json << "\t\t\t" << ids[k];
            if (k + 1 != ids.size())
            {
                json << ",";
            }
            json << "\n";
        }
        json << "\t\t]";
        if (i + 1 != numElem)
        {
            json << ",";
        }
        json << "\n";
    }

    // Nodes
    json << "\t],\n\t\"nodes\":[\n";
    const int numNodes = velocityMesh.nodes.size();
    for (int i = 0; i < numNodes; i++)
    {
        const auto & node = velocityMesh.nodes[i];
        json << "\t\t[\n";
        json << "\t\t\t" << node.x << ",\n";
        json << "\t\t\t" << node.y << "\n";
        json << "\t\t]";
        if (i + 1 != numNodes)
        {
            json << ",";
        }
        json << "\n";
    }

    // Mesh info
    json << "\t],\n";
    json << "\t\"elementsCount\": " << numElem << ",\n";
    json << "\t\"velocityNodesCount\": " << numVelocityNodes << ",\n";
    json << "\t\"pressureNodesCount\": " << numPressureNodes << ",\n";
    json << "\t\"elementSize\": " << 6 << ",\n";


    // Borders
    json << "\t\"uDirichlet\":[\n";
    json << "\t\t{\n";

    auto printBorderNodeIds = [&](const std::vector<int> & v)
    {
        json << "\t\t\t\"nodes\":[\n";
        const int n = v.size();
        for (int i = 0; i < n; i++)
        {
            json << "\t\t\t\t" << v[i];
            if (i + 1 != n)
            {
                json << ",";
            }
            json << "\n";
        }
        json << "\t\t\t],\n";
    };

    printBorderNodeIds(leftNodes);
    json << "\t\t\t\"u\":\"" << peakVelocity << "*y*(0.41-y)\\/(0.41*0.41)\",\n";
    json << "\t\t\t\"v\":\"0\"\n";
    json << "\t\t},\n\t\t{\n";

    printBorderNodeIds(centralNodes);
    json << "\t\t\t\"u\":\"0\",\n";
    json << "\t\t\t\"v\":\"0\"\n";
    json << "\t\t}\n\t],\n";

    json << "\t\"pDirichlet\":[\n";
    json << "\t\t{\n";
    printBorderNodeIds(rightNodes);
    json << "\t\t\t\"p\":\"0\"\n";
    json << "\t\t}\n\t]\n}\n";

    return 0;
}
