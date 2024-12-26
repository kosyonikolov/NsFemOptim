#include <iostream>
#include <string>

#include <mesh/gmsh.h>

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

    dumpPointsAndElements2d("", gmsh.nodeSection.nodes, gmsh.elementSection.elements);
    dumpPhysicalGroups("groups.txt", gmsh.physicsSection);

    return 0;
}
