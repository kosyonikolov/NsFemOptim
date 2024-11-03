#include <iostream>

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./MeshParser <msh file>";
    if (argc != 2)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string meshFileName = argv[1];

    return 0;
}
