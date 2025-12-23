#include <array>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

    int calcCellIdx(const float x, const float y)
    {
        const int ix = std::clamp<int>((x - offsetX) * scaleX, 0, numCellsH - 1);
        const int iy = std::clamp<int>((y - offsetY) * scaleY, 0, numCellsV - 1);
        const int idx = iy * numCellsH + ix;
        return idx;
    }

    std::vector<int> & getCell(const float x, const float y)
    {
        const int idx = calcCellIdx(x, y);
        return cells[idx];
    }

    // Find index of node in mesh that is close enough to pt
    // Assumes that the grid is built on mesh
    int find(const el::Point & pt, const mesh::ConcreteMesh & mesh, const float eps)
    {
        const int bx = std::clamp<int>((pt.x - offsetX) * scaleX, 0, numCellsH - 1);
        const int by = std::clamp<int>((pt.y - offsetY) * scaleY, 0, numCellsV - 1);

        // clang-format off
        constexpr int D = 9;
        const std::array<int, D> dx = {0, -1,  0,  1, -1, 1, -1, 0, 1};
        const std::array<int, D> dy = {0, -1, -1, -1,  0, 0,  1, 1, 1};
        // clang-format on

        // Search in the "expected" cell first, then it its neighbours
        // The point could be there because of numerical imprecision
        for (int i = 0; i < D; i++)
        {
            const int ix = bx + dx[i];
            const int iy = by + dy[i];
            if (ix < 0 || ix >= numCellsH || iy < 0 || iy >= numCellsV)
            {
                continue;
            }

            const int idx = iy * numCellsH + ix;
            for (int j : cells[idx])
            {
                const auto & testPt = mesh.nodes[j];
                const float deltaX = testPt.x - pt.x;
                const float deltaY = testPt.y - pt.y;
                if (std::abs(deltaX) > eps || std::abs(deltaY) > eps)
                {
                    continue;
                }
                return j;
            }
        }

        return -1;
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

el::Type parseElementType(const std::string & name)
{
#define RETIF(x) if (name == #x) return el::Type::x;
    RETIF(P0);
    RETIF(P1);
    RETIF(P2);
#undef RETIF

    throw std::invalid_argument(std::format("Unknown element name [{}]", name));
}

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./MeshParser <element name> <small mesh> <big mesh> <output map file>";
    if (argc != 5)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string elementName = argv[1];
    const std::string smallFileName = argv[2];
    const std::string bigFileName = argv[3];
    const std::string outputFileName = argv[4];

    auto smallTri = mesh::parseTriangleGmsh(smallFileName);
    auto bigTri = mesh::parseTriangleGmsh(bigFileName);

    const auto elementType = parseElementType(elementName);
    const auto baseElement = el::createElement(elementType);

    auto small = mesh::createMesh(smallTri, *baseElement);
    auto big = mesh::createMesh(bigTri, *baseElement);

    // Create grid on big mesh, find each point from the small mesh in it
    Grid grid = createGrid(big);

    const int nSmall = small.nodes.size();
    std::vector<int> result(nSmall, -1);
    const float eps = 1e-6;
    for (int i = 0; i < nSmall; i++)
    {
        const auto refPt = small.nodes[i];
        const int j = grid.find(refPt, big, eps);
        const bool found = j != -1;
        result[i] = j;
        if (!found)
        {
            std::cerr << std::format("Node {} [{}, {}] not found in big mesh!\n", i, refPt.x, refPt.y);
        }
    }

    const int numFound = std::count_if(result.begin(), result.end(), [](int x)
                                       { return x != -1; });
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
