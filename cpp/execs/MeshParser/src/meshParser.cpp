#include <cassert>
#include <format>
#include <iostream>
#include <string>

#include <element/affineTransform.h>
#include <element/factory.h>

#include <mesh/colorScale.h>
#include <mesh/concreteMesh.h>
#include <mesh/drawMesh.h>
#include <mesh/gmsh.h>
#include <mesh/interpolator.h>
#include <mesh/io.h>

template <el::Type t>
void testShapeFunctions()
{
    const auto elem = el::createElement(t);
    const int nDof = elem->dof();
    const auto pts = elem->nodes();
    assert(nDof == pts.size());

    const float eps = 1e-6;
    std::vector<float> shapeVals(nDof);
    for (int i = 0; i < nDof; i++)
    {
        elem->shape(pts[i].x, pts[i].y, shapeVals.data());
        for (int k = 0; k < nDof; k++)
        {
            const float target = k == i ? 1 : 0;
            const float test = shapeVals[k];
            const float delta = test - target;
            if (std::abs(delta) > eps)
            {
                std::cerr << std::format("Shape function fail: i = {} ({}, {}), k = {} , target = {}, actual = {}\n", i,
                                         pts[i].x, pts[i].y, k, target, test);
            }
        }
    }
}

std::vector<float> createInterpolationValues(const mesh::ConcreteMesh & mesh)
{
    const float ox = 1;
    const float oy = 0.2;
    const int n = mesh.nodes.size();
    const float k = 3;
    std::vector<float> result(n);
    for (int i = 0; i < n; i++)
    {
        const float dx = mesh.nodes[i].x - ox;
        const float dy = mesh.nodes[i].y - oy;
        const float r2 = dx * dx + dy * dy;
        result[i] = std::exp(-k * r2);
    }
    return result;
}

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
        std::array<el::Point, 3> pts;
        for (int i = 0; i < 3; i++)
        {
            pts[i] = triMesh.nodes[ptIds[i]];
        }

        std::array<el::Point, 3> refPts = {el::Point{0, 0}, el::Point{1, 0}, el::Point{0, 1}};
        const auto transform = el::calcAffineTransformFromRefTriangle(pts.data());
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

    testShapeFunctions<el::Type::P0>();
    testShapeFunctions<el::Type::P1>();
    testShapeFunctions<el::Type::P2>();

    const auto elementType = el::Type::P2;
    const auto baseElement = el::createElement(elementType);

    {
        const auto nodes = baseElement->nodes();
        for (const auto & node : nodes)
        {
            std::cout << std::format("{} {}\n", node.x, node.y);
        }
    }

    auto mesh = mesh::createMesh(triMesh, *baseElement);

    if (true)
    {
        std::cout << "Testing if mesh is proper\n";

        assert(mesh.numElements == triMesh.elements.size());
        const int nElements = mesh.numElements;

        const auto refPts = mesh.baseElement->nodes();
        const int elSize = refPts.size();
        assert(elSize == mesh.getElementSize());

        std::vector<int> ids(elSize);
        std::vector<el::Point> targetPts(elSize);

        const float eps = 1e-5f;
        int failedCount = 0;

        for (int i = 0; i < nElements; i++)
        {
            mesh.getElement(i, ids.data(), targetPts.data());

            // Ref coords -> global coords
            const auto transform = mesh.elementTransforms[i];
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

            // Global coords -> ref coords
            const auto invTransform = mesh.invElementTransforms[i];
            for (int k = 0; k < elSize; k++)
            {
                const auto test = invTransform(targetPts[k]);
                const auto target = refPts[k];
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

    if (true)
    {
        const cv::Mat img = mesh::drawMesh(mesh, 3500);
        cv::imwrite("mesh.png", img);
    }

    if (true)
    {
        // Test interpolator
        mesh::Interpolator interp(mesh, 0.05);
        const int numNodes = mesh.nodes.size();
        std::vector<float> values = createInterpolationValues(mesh);
        interp.setValues(values);

        for (int i = 0; i < numNodes; i++)
        {
            const auto pt = mesh.nodes[i];
            const float target = values[i];
            const auto test = interp.interpolate(pt.x, pt.y);
            if (!test)
            {
                std::cerr << std::format("Failed to interpolate point ({}, {})\n", pt.x, pt.y);
                continue;
            }

            const float v = test.value();
            const float delta = v - target;
            if (std::abs(delta) > 1e-6f)
            {
                std::cerr << std::format("Bad interpolation on ({}, {}) - expected {}, got {} (delta = {})\n", pt.x,
                                         pt.y, target, v, delta);
            }
        }

        mesh::SimpleColorScale scc(0, 1,
                                   {cv::Scalar(128, 0, 0), cv::Scalar(0, 0, 128), cv::Scalar(0, 255, 255)});

        cv::Mat valImg = mesh::drawValues(interp, scc, 1000);
        cv::imwrite("values.png", valImg);
    }

    return 0;
}
