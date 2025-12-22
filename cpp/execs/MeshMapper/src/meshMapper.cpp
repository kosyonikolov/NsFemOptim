#include <format>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <element/affineTransform.h>
#include <element/factory.h>

#include <mesh/colorScale.h>
#include <mesh/concreteMesh.h>
#include <mesh/drawMesh.h>
#include <mesh/gmsh.h>
#include <mesh/interpolator.h>
#include <mesh/io.h>

struct Grid
{
    int numCellsH, numCellsV;
    float offsetX, offsetY;
    float scaleX, scaleY;
    std::vector<std::vector<int>> cells;

    std::vector<int> & getCell(const float x, const float y)
    {
        const int ix = std::clamp<int>((x - offsetX) * scaleX, 0, numCellsH - 1);
        const int iy = std::clamp<int>((y - offsetY) * scaleY, 0, numCellsV - 1);
        const int idx = iy * numCellsH + ix;
        return cells[idx];
    }
};

Grid createGrid(const mesh::ConcreteMesh & mesh)
{
    // Find min/max X and Y values
    float minX = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::min();
    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::min();

    const int numNodes = mesh.nodes.size();
    for (int i = 0; i < numNodes; i++)
    {
        const auto pt = mesh.nodes[i];
        minX = std::min(pt.x, minX);
        maxX = std::max(pt.x, maxX);
        minY = std::min(pt.y, minY);
        maxY = std::max(pt.y, maxY);
    }

    const float width = maxX - minX;
    const float height = maxY - minY;

    const int maxNumCells = 1000000;
    const int desiredNumCells = std::min(numNodes, maxNumCells);

    // N = (width / h) * (height / h) = (width * height) / h^2
    // <=> h^2 = (width * height) / N <=> h = sqrt((width * height) / N) 
    const float cellSize = std::sqrt(width * height / desiredNumCells);
    
    Grid result;
    result.numCellsH = std::max<int>(1, width / cellSize);
    result.numCellsV = std::max<int>(1, height / cellSize);
    result.offsetX = minX;
    result.scaleX = (result.numCellsH - 1) / width;
    result.offsetY = minY;
    result.scaleY = (result.numCellsV - 1) / height;

    result.cells.resize(result.numCellsH * result.numCellsV);
    for (int i = 0; i < numNodes; i++)
    {
        const auto pt = mesh.nodes[i];
        auto & cell = result.getCell(pt.x, pt.y);
        cell.push_back(i);
    }

    return result;
}

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./MeshParser <small mesh> <big mesh> <output map file>";
    if (argc != 4)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string smallFileName = argv[1];
    const std::string bigFileName = argv[2];
    const std::string outputFileName = argv[3];

    auto smallTri = mesh::parseTriangleGmsh(smallFileName);
    auto bigTri = mesh::parseTriangleGmsh(bigFileName);

    const auto elementType = el::Type::P2;
    const auto baseElement = el::createElement(elementType);

    auto small = mesh::createMesh(smallTri, *baseElement);
    auto big = mesh::createMesh(bigTri, *baseElement);

    // Create grid on big mesh, find each point from the small mesh in it
    Grid grid = createGrid(big);

    const int nSmall = small.nodes.size();
    std::vector<int> result(nSmall, -1);
    const float eps = 1e-6;
    int numFound = 0;
    for (int i = 0; i < nSmall; i++)
    {
        const auto refPt = small.nodes[i];
        bool found = false;
        auto & cell = grid.getCell(refPt.x, refPt.y);
        for (int j : cell)
        {
            const auto testPt = big.nodes[j];
            const float dx = refPt.x - testPt.x;
            const float dy = refPt.y - testPt.y;
            if (std::abs(dx) < eps && std::abs(dy) < eps)
            {
                found = true;
                result[i] = j;
                numFound++;
                break;
            }
        }
        if (!found)
        {
            std::cerr << std::format("Node {} [{}, {}] not found in big mesh!\n", i, refPt.x, refPt.y);
        }
    }

    std::cout << std::format("Found points: {} / {}\n", numFound, nSmall);
    
    std::ofstream file(outputFileName);
    if (!file.is_open())
    {
        std::cerr << std::format("Failed to open output file [{}]\n", outputFileName);
        return 1;
    }

    for (int k : result)
    {
        file << k << "\n";
    }

    return 0;
}
