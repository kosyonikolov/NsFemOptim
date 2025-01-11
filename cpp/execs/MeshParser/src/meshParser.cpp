#include <iostream>
#include <string>

#include <mesh/gmsh.h>
#include <mesh/io.h>
#include <mesh/concreteMesh.h>

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

    const auto elementType = mesh::ElementType::P2;
    const auto baseElement = mesh::createElement(elementType); 
    
    auto mesh = mesh::createMesh(triMesh, baseElement);

    return 0;
}
