#include <iostream>
#include <string>
#include <format>

#include <mesh/gmsh.h>
#include <mesh/io.h>
#include <mesh/concreteMesh.h>
#include <mesh/affineTransform.h>

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

        std::array<mesh::Point, 3> refPts = {mesh::Point{0,0}, mesh::Point{1,0}, mesh::Point{0,1}};
        const auto transform = mesh::calcAffineTransformFromRefTriangle(pts.data());
        for (int i = 0; i < 3; i++)
        {
            const auto test = transform(refPts[i]);
            const auto target = pts[i];
            const float dx = test.x - target.x;
            const float dy = test.y - target.y;
            std::cout << std::format("Target = ({}, {}), test = ({}, {}), delta = ({}, {})\n",
                                     target.x, target.y,
                                     test.x, test.y,
                                     dx, dy);
        }
    }

    const auto elementType = mesh::ElementType::P2;
    const auto baseElement = mesh::createElement(elementType); 
    
    auto mesh = mesh::createMesh(triMesh, baseElement);

    return 0;
}
