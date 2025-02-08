#include <cassert>
#include <format>
#include <iostream>
#include <string>

#include <Eigen/Sparse>

#include <element/affineTransform.h>
#include <element/factory.h>
#include <element/triangleIntegrator.h>

#include <mesh/colorScale.h>
#include <mesh/concreteMesh.h>
#include <mesh/drawMesh.h>
#include <mesh/gmsh.h>
#include <mesh/interpolator.h>
#include <mesh/io.h>

float testFunction(const float x, const float y)
{
    const float ox = 1;
    const float oy = 0.2;
    const float k = 1;

    const float dx = x - ox;
    const float dy = y - oy;
    const float r2 = dx * dx + dy * dy;

    const float s = std::sin(x * 20);

    return std::exp(-k * r2) * s * s;
}

std::vector<float> createInterpolationValues(const mesh::ConcreteMesh & mesh)
{
    const int n = mesh.nodes.size();
    std::vector<float> result(n);
    for (int i = 0; i < n; i++)
    {
        result[i] = testFunction(mesh.nodes[i].x, mesh.nodes[i].y);
    }
    return result;
}

std::vector<float> projectValues(const mesh::ConcreteMesh & mesh)
{
    el::TriangleIntegrator integrator(mesh.baseElement, 4);
    const int numNodes = mesh.nodes.size();
    const int nElem = mesh.numElements;
    const int elSize = mesh.getElementSize();
    using SolType = double;
    using SpMat = Eigen::SparseMatrix<SolType>;
    using Triplet = Eigen::Triplet<SolType>;

    SpMat m(numNodes, numNodes);
    Eigen::Vector<SolType, Eigen::Dynamic> rhs(numNodes);
    rhs.setZero();

    // Assemble
    std::vector<Triplet> triplets;
    
    std::vector<int> ids(elSize);
    std::vector<el::Point> pts(elSize);
    cv::Mat localMassMatrix;
    std::vector<float> localLoadVector;
    for (int i = 0; i < nElem; i++)
    {
        mesh.getElement(i, ids.data(), pts.data());
        const auto t = mesh.elementTransforms[i];

        integrator.integrateLocalMassMatrix(t, localMassMatrix);
        integrator.integrateLocalLoadVector(t, testFunction, localLoadVector);
        assert(elSize == localMassMatrix.cols);
        assert(elSize == localMassMatrix.rows);
        assert(elSize == localLoadVector.size());

        // Accumulate
        for (int r = 0; r < elSize; r++)
        {
            const int i = ids[r];
            for (int c = 0; c < elSize; c++)
            {
                const int j = ids[c];
                const Triplet triplet(i, j, localMassMatrix.at<float>(r, c));
                triplets.push_back(triplet);
            }

            rhs[i] += localLoadVector[r];
        }
    }

    // Build matrix
    m.setFromTriplets(triplets.begin(), triplets.end());

    // Solve
    Eigen::SimplicialLDLT<SpMat> solver(m);
    Eigen::Vector<SolType, Eigen::Dynamic> sol = solver.solve(rhs);

    std::vector<float> result(numNodes);
    for (int i = 0; i < numNodes; i++)
    {
        result[i] = sol[i];
    }

    return result;
}

float measureError(const mesh::Interpolator & interp)
{
    const auto [minX, minY, width, height] = interp.getRange();
    const float h = 0.005;
    const int cols = width / h;
    const int rows = height / h;

    double sum = 0;
    int count = 0;
    for (int iy = 0; iy <= rows; iy++)
    {
        const float y = minY + iy * height / rows;
        for (int ix = 0; ix <= cols; ix++)
        {   
            const float x = minX + ix * width / cols;
            auto iVal = interp.interpolate(x, y);
            if (!iVal)
            {
                continue;
            }

            const float target = testFunction(x, y);
            const float delta = iVal.value() - target;
            sum += delta * delta;
            count++;
        }
    }

    const float avg = std::sqrt(sum / count);
    return avg;
}

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./L2Projection <msh file>";
    if (argc != 2)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string meshFileName = argv[1];

    auto gmsh = mesh::parseGmsh(meshFileName);
    auto triMesh = mesh::parseTriangleGmsh(gmsh);

    const auto elementType = el::Type::P2;
    const auto baseElement = el::createElement(elementType);

    auto mesh = mesh::createMesh(triMesh, *baseElement);

    std::vector<float> projValues = projectValues(mesh);

    mesh::Interpolator interp(mesh, 0.05);
    mesh::SimpleColorScale scc(0, 1,
                               {cv::Scalar(128, 0, 0), cv::Scalar(0, 0, 128), cv::Scalar(0, 255, 255)});

    interp.setValues(projValues);
    const float projErr = measureError(interp);
    std::cout << std::format("L2 projection error: {}\n", projErr);
    cv::Mat outImg = mesh::drawValues(interp, scc, 1000);
    cv::imwrite("values_proj.png", outImg);

    if (true)
    {
        const cv::Mat img = mesh::drawMesh(mesh, 3500);
        cv::imwrite("mesh.png", img);
    }

    if (true)
    {
        // Draw interpolated image
        std::vector<float> values = createInterpolationValues(mesh);
        interp.setValues(values);

        const float interpErr = measureError(interp);
        std::cout << std::format("Interpolation error: {}\n", interpErr);

        cv::Mat valImg = mesh::drawValues(interp, scc, 1000);
        cv::imwrite("values_interp.png", valImg);
    }

    return 0;
}
