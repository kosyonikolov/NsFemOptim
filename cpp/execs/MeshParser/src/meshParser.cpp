#include <cassert>
#include <format>
#include <iostream>
#include <string>

#include <mesh/affineTransform.h>
#include <mesh/concreteMesh.h>
#include <mesh/gmsh.h>
#include <mesh/io.h>
#include <mesh/drawMesh.h>

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./MeshParser <msh file>";
    if (argc != 2)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string meshFileName = argv[1];

    auto gmsh = mesh::parseGmsh(meshFileName);

    // dumpPointsAndElements2d("", gmsh.nodeSection.nodes, gmsh.elementSection.elements);
    // dumpPhysicalGroups("groups.txt", gmsh.physicsSection);

    auto triMesh = mesh::parseTriangleGmsh(gmsh);

    if (false)
    {
        // Test affine transform
        auto ptIds = triMesh.elements[0];
        std::array<mesh::Point, 3> pts;
        for (int i = 0; i < 3; i++)
        {
            pts[i] = triMesh.nodes[ptIds[i]];
        }

        std::array<mesh::Point, 3> refPts = {mesh::Point{0, 0}, mesh::Point{1, 0}, mesh::Point{0, 1}};
        const auto transform = mesh::calcAffineTransformFromRefTriangle(pts.data());
        for (int i = 0; i < 3; i++)
        {
            const auto test = transform(refPts[i]);
            const auto target = pts[i];
            const float dx = test.x - target.x;
            const float dy = test.y - target.y;
            std::cout << std::format("Target = ({}, {}), test = ({}, {}), delta = ({}, {})\n", target.x, target.y,
                                     test.x, test.y, dx, dy);
        }
    }

    const auto elementType = mesh::ElementType::P2;
    const auto baseElement = mesh::createElement(elementType);

    auto mesh = mesh::createMesh(triMesh, baseElement);

    if (true)
    {
        std::cout << "Testing if mesh is proper\n";

        assert(mesh.numElements == triMesh.elements.size());
        const int nElements = mesh.numElements;

        const auto refPts = mesh.baseElement.getAllNodes();
        const int elSize = refPts.size();
        assert(elSize == mesh.getElementSize());

        std::vector<int> ids(elSize);
        std::vector<mesh::Point> targetPts(elSize);

        const float eps = 1e-6f;
        int failedCount = 0;

        for (int i = 0; i < nElements; i++)
        {
            const auto transform = mesh.elementTransforms[i];
            mesh.getElement(i, ids.data(), targetPts.data());

            for (int k = 0; k < elSize; k++)
            {
                const auto test = transform(refPts[k]);
                const auto target = targetPts[k];
                const float dx = test.x - target.x;
                const float dy = test.y - target.y;
                if (std::abs(dx) >= eps || std::abs(dy) >= eps)
                {
                    std::cout << std::format("Target = ({}, {}), test = ({}, {}), delta = ({}, {})\n", target.x,
                                             target.y, test.x, test.y, dx, dy);
                    failedCount++;
                }
            }
        }

        std::cout << std::format("Failed points: {}\n", failedCount);
    }

    if (false)
    {
        // Print border elements
        const int nBorder = mesh.numBorderElements;
        const int borderElSize = mesh.getBorderElementSize();
        std::vector<int> ids(borderElSize);
        for (int i = 0; i < nBorder; i++)
        {
            int triId, side, group;
            mesh.getBorderElement(i, triId, side, group, ids.data(), 0);
            std::cout << std::format("Border element {}: tri = {}, side = {}, group = {}, ids = [", i, triId, side,
                                     group);

            for (int j = 0; j < borderElSize; j++)
            {
                std::cout << ids[j];
                if (j != borderElSize - 1)
                {
                    std::cout << " ";
                }
            }
            std::cout << "]\n";
        }
    }

    const cv::Mat img = mesh::drawMesh(mesh, 3500);
    cv::imwrite("mesh.png", img);

    return 0;
}
