#include <cassert>
#include <format>
#include <iostream>
#include <string>

#include <Eigen/Sparse>

#include <element/affineTransform.h>
#include <element/calc.h>
#include <element/triangleIntegrator.h>

#include <mesh/colorScale.h>
#include <mesh/concreteMesh.h>
#include <mesh/drawMesh.h>
#include <mesh/gmsh.h>
#include <mesh/interpolator.h>
#include <mesh/io.h>

using SolType = double;
using SpMat = Eigen::SparseMatrix<SolType>;
using Triplet = Eigen::Triplet<SolType>;
using Vector = Eigen::Vector<SolType, Eigen::Dynamic>;

struct DirichletNode
{
    int id;
    float value;
    bool operator<(const DirichletNode & other) const
    {
        return id < other.id;
    }
};

std::vector<DirichletNode> extractDirichletNodesSimple(const mesh::ConcreteMesh & mesh,
                                                       const std::vector<int> borderIds,
                                                       const std::vector<float> borderValues)
{
    const int n = borderIds.size();
    if (n != borderValues.size())
    {
        throw std::invalid_argument(std::format("{}: Ids/values have different sizes", __FUNCTION__));
    }

    std::vector<DirichletNode> result;

    const int elSize = mesh.getBorderElementSize();
    const int numBorderElems = mesh.numBorderElements;
    std::vector<int> ptIds(elSize);
    std::vector<bool> seen(mesh.nodes.size(), false);
    for (int i = 0; i < numBorderElems; i++)
    {
        int triangle, side, group;
        mesh.getBorderElement(i, triangle, side, group, ptIds.data(), 0);
        auto it = std::find(borderIds.begin(), borderIds.end(), group);
        if (it == borderIds.end())
        {
            continue;
        }
        const int j = it - borderIds.begin();
        const float val = borderValues[j];
        for (int k = 0; k < elSize; k++)
        {
            const int nodeIdx = ptIds[k];
            if (seen[nodeIdx])
            {
                continue;
            }
            seen[nodeIdx] = true;
            result.push_back(DirichletNode{ptIds[k], val});
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

std::vector<int> extractInternalNodes(const int numNodes, const std::vector<DirichletNode> & sortedDirichletNodes)
{
    assert(std::is_sorted(sortedDirichletNodes.begin(), sortedDirichletNodes.end()));
    std::vector<int> result;

    int i = 0;
    int j = 0;
    while (i < numNodes && j < sortedDirichletNodes.size())
    {
        const int d = sortedDirichletNodes[j].id;
        if (i < d)
        {
            result.push_back(i);
            i++;
        }
        else if (i == d)
        {
            i++;
            j++;
        }
        else if (d < i)
        {
            j++;
        }
    }

    while (i < numNodes)
    {
        result.push_back(i);
        i++;
    }

    return result;
}

std::vector<Triplet> projectTriplets(const int numNodes, const std::vector<Triplet> & orig, const std::vector<int> & newIds)
{
    std::vector<int> remap(numNodes, -1); // Old to new idx
    for (int i = 0; i < newIds.size(); i++)
    {
        const int j = newIds[i];
        remap[j] = i;
    }

    std::vector<Triplet> result;
    for (const Triplet & t : orig)
    {
        const int i = remap[t.row()];
        const int j = remap[t.col()];
        const auto v = t.value();
        if (i < 0 || j < 0)
        {
            continue;
        }
        result.push_back(Triplet(i, j, v));
    }

    return result;
};

std::vector<float> solveHeatEquation(const mesh::ConcreteMesh & mesh)
{
    const int idLeft = mesh.findGroupId("Left");
    if (idLeft < 0)
    {
        throw std::invalid_argument("No left border!");
    }

    const int idRight = mesh.findGroupId("Right");
    if (idRight < 0)
    {
        throw std::invalid_argument("No right border!");
    }

    const int idCircle = mesh.findGroupId("Circle");
    if (idCircle < 0)
    {
        throw std::invalid_argument("No circle border!");
    }

    const float leftVal = 0;
    const float rightVal = 1;
    const float circleVal = 10;

    const int numNodes = mesh.nodes.size();
    const std::vector<int> borderIds{idLeft, idRight, idCircle};
    const std::vector<float> borderValues{leftVal, rightVal, circleVal};
    const auto dirichletNodes = extractDirichletNodesSimple(mesh, borderIds, borderValues);
    const auto internalNodes = extractInternalNodes(numNodes, dirichletNodes);

    {
        // Sanity check - the union of the Dirichlet and internal nodes should give all nodes
        auto all = internalNodes;
        for (const auto & dn : dirichletNodes)
        {
            all.push_back(dn.id);
        }
        std::sort(all.begin(), all.end());
        const int n = all.size();
        assert(n == numNodes);
        for (int i = 0; i < numNodes; i++)
        {
            assert(all[i] == i);
        }
    }

    el::TriangleIntegrator integrator(mesh.baseElement, 4);
    const int nElem = mesh.numElements;
    const int elSize = mesh.getElementSize();

    // {
    //     // Test integrator
    //     cv::Mat m1;
    //     auto identity = el::AffineTransform::identity();
    //     integrator.integrateLocalStiffnessMatrix(identity, m1);
    //     std::cout << m1 << "\n";
    // }

    SpMat m(numNodes, numNodes);

    // Assemble
    std::vector<Triplet> triplets;

    std::vector<int> ids(elSize);
    std::vector<el::Point> pts(elSize);
    cv::Mat localStiffnessMatrix;
    for (int i = 0; i < nElem; i++)
    {
        mesh.getElement(i, ids.data(), pts.data());
        const auto t = mesh.elementTransforms[i];

        // std::cout << std::format("Element {}:\nPoints:\n", i);
        // for (const auto & pt : pts)
        // {
        //     std::cout << pt.x << " " << pt.y << "\n";
        // }

        integrator.integrateLocalStiffnessMatrix(t, localStiffnessMatrix);
        assert(elSize == localStiffnessMatrix.cols);
        assert(elSize == localStiffnessMatrix.rows);

        // std::cout << "Local matrix:\n";
        // std::cout << localStiffnessMatrix << "\n";

        // Accumulate
        for (int r = 0; r < elSize; r++)
        {
            const int i = ids[r];
            for (int c = 0; c < elSize; c++)
            {
                const int j = ids[c];
                const Triplet triplet(i, j, localStiffnessMatrix.at<float>(r, c));
                triplets.push_back(triplet);
            }
        }
    }

    // Build matrix
    m.setFromTriplets(triplets.begin(), triplets.end());

    // Handle dirichlet nodes
    Vector b0(numNodes);
    b0.setZero();
    for (const auto [id, val] : dirichletNodes)
    {
        b0[id] = val;
    }

    // Compose system for internal nodes
    const int numInternal = internalNodes.size();
    Vector sub = m * b0;
    Vector bInternal(numInternal);
    const auto internalTriplets = projectTriplets(numNodes, triplets, internalNodes);
    SpMat mInternal(numInternal, numInternal);
    mInternal.setFromTriplets(internalTriplets.begin(), internalTriplets.end());
    for (int i = 0; i < numInternal; i++)
    {
        const int j = internalNodes[i];
        bInternal[i] = -sub[j];
    }

    // Solve for internal nodes
    Eigen::SimplicialLDLT<SpMat> solver(mInternal);
    Eigen::Vector<SolType, Eigen::Dynamic> qInt = solver.solve(bInternal);

    std::vector<float> result(numNodes);
    for (int i = 0; i < numInternal; i++)
    {
        const int j = internalNodes[i];
        result[j] = qInt[i];
    }
    for (const auto [j, value] : dirichletNodes)
    {
        result[j] = value;
    }

    return result;
}

// Neumann (x1) and Dirichlet (x2) conditions
std::vector<float> solveHeatEquation2(const mesh::ConcreteMesh & mesh)
{
    const int idLeft = mesh.findGroupId("Left");
    if (idLeft < 0)
    {
        throw std::invalid_argument("No left border!");
    }

    const int idRight = mesh.findGroupId("Right");
    if (idRight < 0)
    {
        throw std::invalid_argument("No right border!");
    }

    const int idCircle = mesh.findGroupId("Circle");
    if (idCircle < 0)
    {
        throw std::invalid_argument("No circle border!");
    }

    const float rightVal = 10;
    const float circleVal = -50;

    const int numNodes = mesh.nodes.size();
    const std::vector<int> borderIds{ idRight, idCircle };
    const std::vector<float> borderValues{ rightVal, circleVal };
    const auto dirichletNodes = extractDirichletNodesSimple(mesh, borderIds, borderValues);
    const auto internalNodes = extractInternalNodes(numNodes, dirichletNodes);

    auto flow = [](const el::Point & pt) -> el::Point
    {
        const float s = (pt.y - 0.2);
        el::Point r;
        // r.x = -0.001 * (0.3f - 4.5f * s*s);
        r.x = -1000 * std::exp(-100 * s * s);
        r.y = 0;
        return r;
    };

    el::TriangleIntegrator integrator(mesh.baseElement, 4);
    const int nElem = mesh.numElements;
    const int elSize = mesh.getElementSize();

    SpMat m(numNodes, numNodes);

    // Assemble
    std::vector<Triplet> triplets;

    std::vector<int> ids(elSize);
    std::vector<el::Point> pts(elSize);
    cv::Mat localStiffnessMatrix;
    for (int i = 0; i < nElem; i++)
    {
        mesh.getElement(i, ids.data(), pts.data());
        const auto t = mesh.elementTransforms[i];

        integrator.integrateLocalStiffnessMatrix(t, localStiffnessMatrix);
        assert(elSize == localStiffnessMatrix.cols);
        assert(elSize == localStiffnessMatrix.rows);

        // Accumulate
        for (int r = 0; r < elSize; r++)
        {
            const int i = ids[r];
            for (int c = 0; c < elSize; c++)
            {
                const int j = ids[c];
                const Triplet triplet(i, j, localStiffnessMatrix.at<float>(r, c));
                triplets.push_back(triplet);
            }
        }
    }

    // Build matrix
    m.setFromTriplets(triplets.begin(), triplets.end());

    // Build vector
    Vector b0(numNodes);
    b0.setZero();

    const int numBorderElements = mesh.numBorderElements;
    std::vector<float> localLoadVector;
    for (int i = 0; i < numBorderElements; i++)
    {
        int triangleId, side, group;
        mesh.getBorderElement(i, triangleId, side, group, 0, 0);
        if (group != idLeft)
        {
            continue;
        }
        mesh.getElement(triangleId, ids.data(), 0);

        const auto & t = mesh.elementTransforms[triangleId];
        integrator.integrateLocalBorderLoadVector(t, flow, side, localLoadVector);
        
        // Accumulate
        assert(localLoadVector.size() == elSize);
        for (int k = 0; k < elSize; k++)
        {
            const int j = ids[k];
            b0[j] += localLoadVector[k];
        }
    }

    // Handle dirichlet nodes
    Vector d(numNodes);
    d.setZero();
    for (const auto [id, val] : dirichletNodes)
    {
        d[id] = val;
    }

    // Compose system for internal nodes
    const int numInternal = internalNodes.size();
    Vector sub = m * d;
    Vector bInternal(numInternal);
    const auto internalTriplets = projectTriplets(numNodes, triplets, internalNodes);
    SpMat mInternal(numInternal, numInternal);
    mInternal.setFromTriplets(internalTriplets.begin(), internalTriplets.end());
    for (int i = 0; i < numInternal; i++)
    {
        const int j = internalNodes[i];
        bInternal[i] = b0[j] - sub[j];
    }

    // Solve for internal nodes
    Eigen::SimplicialLDLT<SpMat> solver(mInternal);
    Eigen::Vector<SolType, Eigen::Dynamic> qInt = solver.solve(bInternal);

    std::vector<float> result(numNodes);
    for (int i = 0; i < numInternal; i++)
    {
        const int j = internalNodes[i];
        result[j] = qInt[i];
    }
    for (const auto [j, value] : dirichletNodes)
    {
        result[j] = value;
    }

    return result;
}

// Neumann (x2) and Dirichlet (x1) conditions
std::vector<float> solveHeatEquation3(const mesh::ConcreteMesh & mesh)
{
    const int idLeft = mesh.findGroupId("Left");
    if (idLeft < 0)
    {
        throw std::invalid_argument("No left border!");
    }

    const int idRight = mesh.findGroupId("Right");
    if (idRight < 0)
    {
        throw std::invalid_argument("No right border!");
    }

    const int idCircle = mesh.findGroupId("Circle");
    if (idCircle < 0)
    {
        throw std::invalid_argument("No circle border!");
    }

    const float rightVal = 10;

    const int numNodes = mesh.nodes.size();
    const std::vector<int> borderIds{ idRight };
    const std::vector<float> borderValues{ rightVal };
    const auto dirichletNodes = extractDirichletNodesSimple(mesh, borderIds, borderValues);
    const auto internalNodes = extractInternalNodes(numNodes, dirichletNodes);

    auto flow = [](const el::Point & /*pt*/) -> el::Point
    {
        // const float s = (pt.y - 0.2);
        el::Point r;
        // r.x = -0.001 * (0.3f - 4.5f * s*s);
        // r.x = -17 * std::exp(-90 * s * s);
        r.x = -8;
        r.y = 0;
        return r;
    };

    auto circleFlow = [](const el::Point & pt) -> el::Point
    {
        const el::Point center {0.2, 0.2};
        const float dx = pt.x - center.x;
        const float dy = pt.y - center.y;
        const el::Point normal = el::normalize(el::Point{dx, dy});
        const float k = 12;
        return {k * normal.x, k * normal.y};
    };


    el::TriangleIntegrator integrator(mesh.baseElement, 4);
    const int nElem = mesh.numElements;
    const int elSize = mesh.getElementSize();

    SpMat m(numNodes, numNodes);

    // Assemble
    std::vector<Triplet> triplets;

    std::vector<int> ids(elSize);
    std::vector<el::Point> pts(elSize);
    cv::Mat localStiffnessMatrix;
    for (int i = 0; i < nElem; i++)
    {
        mesh.getElement(i, ids.data(), pts.data());
        const auto t = mesh.elementTransforms[i];

        integrator.integrateLocalStiffnessMatrix(t, localStiffnessMatrix);
        assert(elSize == localStiffnessMatrix.cols);
        assert(elSize == localStiffnessMatrix.rows);

        // Accumulate
        for (int r = 0; r < elSize; r++)
        {
            const int i = ids[r];
            for (int c = 0; c < elSize; c++)
            {
                const int j = ids[c];
                const Triplet triplet(i, j, localStiffnessMatrix.at<float>(r, c));
                triplets.push_back(triplet);
            }
        }
    }

    // Build matrix
    m.setFromTriplets(triplets.begin(), triplets.end());

    // Build vector
    Vector b0(numNodes);
    b0.setZero();

    const int numBorderElements = mesh.numBorderElements;
    std::vector<float> localLoadVector;
    for (int i = 0; i < numBorderElements; i++)
    {
        int triangleId, side, group;
        mesh.getBorderElement(i, triangleId, side, group, 0, 0);
        
        if (group != idLeft && group != idCircle)
        {
            continue;
        }
        mesh.getElement(triangleId, ids.data(), pts.data());

        const auto & t = mesh.elementTransforms[triangleId];
        integrator.integrateLocalBorderLoadVector(t, group == idCircle ? circleFlow : flow, side, localLoadVector);

        // Accumulate
        assert(localLoadVector.size() == elSize);
        for (int k = 0; k < elSize; k++)
        {
            const int j = ids[k];
            b0[j] += localLoadVector[k];
        }
    }

    // Handle dirichlet nodes
    Vector d(numNodes);
    d.setZero();
    for (const auto [id, val] : dirichletNodes)
    {
        d[id] = val;
    }

    // Compose system for internal nodes
    const int numInternal = internalNodes.size();
    Vector sub = m * d;
    Vector bInternal(numInternal);
    const auto internalTriplets = projectTriplets(numNodes, triplets, internalNodes);
    SpMat mInternal(numInternal, numInternal);
    mInternal.setFromTriplets(internalTriplets.begin(), internalTriplets.end());
    for (int i = 0; i < numInternal; i++)
    {
        const int j = internalNodes[i];
        bInternal[i] = b0[j] - sub[j];
    }

    // Solve for internal nodes
    Eigen::SimplicialLDLT<SpMat> solver(mInternal);
    Eigen::Vector<SolType, Eigen::Dynamic> qInt = solver.solve(bInternal);

    std::vector<float> result(numNodes);
    for (int i = 0; i < numInternal; i++)
    {
        const int j = internalNodes[i];
        result[j] = qInt[i];
    }
    for (const auto [j, value] : dirichletNodes)
    {
        result[j] = value;
    }

    return result;
}

// Dirichlet BCs and convection
std::vector<float> solveHeatEquation4(const mesh::ConcreteMesh & mesh)
{
    const int idLeft = mesh.findGroupId("Left");
    if (idLeft < 0)
    {
        throw std::invalid_argument("No left border!");
    }

    const int idRight = mesh.findGroupId("Right");
    if (idRight < 0)
    {
        throw std::invalid_argument("No right border!");
    }

    const int idCircle = mesh.findGroupId("Circle");
    if (idCircle < 0)
    {
        throw std::invalid_argument("No circle border!");
    }

    const float leftVal = 0;
    const float rightVal = 1;
    const float circleVal = 10;

    // auto flow = [](const el::Point & pt) -> el::Point
    // {
    //     const float minX = 0.4;
    //     const float maxX = 0.7;
    //     const float k = std::clamp<float>((pt.x - minX) / (maxX - minX), 0, 1);
    //     el::Point flow;
    //     flow.x = -10.6 * k;
    //     flow.y = 0;
    //     return flow;
    // };

    // auto flow = [](const el::Point & pt) -> el::Point
    // {
    //     const float d = pt.y - 0.2;
    //     const float minD = 0.07;
    //     const float maxD = 0.1;
    //     const float k = std::clamp<float>((std::abs(d) - minD) / (maxD - minD), 0, 1);
    //     el::Point flow;
    //     flow.x = 0.8 * k;
    //     flow.y = 0;
    //     return flow;
    // };

    auto flow = [](const el::Point &) -> el::Point
    {
        el::Point flow;
        flow.x = 0;
        flow.y = 0.5;
        return flow;
    };

    const int numNodes = mesh.nodes.size();
    const std::vector<int> borderIds{idLeft, idRight, idCircle};
    const std::vector<float> borderValues{leftVal, rightVal, circleVal};
    const auto dirichletNodes = extractDirichletNodesSimple(mesh, borderIds, borderValues);
    const auto internalNodes = extractInternalNodes(numNodes, dirichletNodes);

    el::TriangleIntegrator integrator(mesh.baseElement, 4);
    const int nElem = mesh.numElements;
    const int elSize = mesh.getElementSize();

    SpMat m(numNodes, numNodes);

    // Assemble
    std::vector<Triplet> triplets;

    std::vector<int> ids(elSize);
    std::vector<el::Point> pts(elSize);
    cv::Mat localStiffnessMatrix;
    cv::Mat localConvectionMatrix;
    for (int i = 0; i < nElem; i++)
    {
        mesh.getElement(i, ids.data(), pts.data());
        const auto t = mesh.elementTransforms[i];

        // std::cout << std::format("Element {}:\nPoints:\n", i);
        // for (const auto & pt : pts)
        // {
        //     std::cout << pt.x << " " << pt.y << "\n";
        // }

        integrator.integrateLocalStiffnessMatrix(t, localStiffnessMatrix);
        assert(elSize == localStiffnessMatrix.cols);
        assert(elSize == localStiffnessMatrix.rows);
        
        integrator.integrateLocalConvectionMatrix(t, flow, localConvectionMatrix);
        assert(elSize == localConvectionMatrix.cols);
        assert(elSize == localConvectionMatrix.rows);

        // std::cout << "Local matrix:\n";
        // std::cout << localStiffnessMatrix << "\n";

        // Accumulate
        for (int r = 0; r < elSize; r++)
        {
            const int i = ids[r];
            for (int c = 0; c < elSize; c++)
            {
                const int j = ids[c];
                const SolType val = localStiffnessMatrix.at<float>(r, c) + localConvectionMatrix.at<float>(r, c);
                const Triplet triplet(i, j, val);
                triplets.push_back(triplet);
            }
        }
    }

    // Build matrix
    m.setFromTriplets(triplets.begin(), triplets.end());

    // Handle dirichlet nodes
    Vector b0(numNodes);
    b0.setZero();
    for (const auto [id, val] : dirichletNodes)
    {
        b0[id] = val;
    }

    // Compose system for internal nodes
    const int numInternal = internalNodes.size();
    Vector sub = m * b0;
    Vector bInternal(numInternal);
    const auto internalTriplets = projectTriplets(numNodes, triplets, internalNodes);
    SpMat mInternal(numInternal, numInternal);
    mInternal.setFromTriplets(internalTriplets.begin(), internalTriplets.end());
    for (int i = 0; i < numInternal; i++)
    {
        const int j = internalNodes[i];
        bInternal[i] = -sub[j];
    }

    // Solve for internal nodes
    Eigen::SimplicialLDLT<SpMat> solver(mInternal);
    Eigen::Vector<SolType, Eigen::Dynamic> qInt = solver.solve(bInternal);

    std::vector<float> result(numNodes);
    for (int i = 0; i < numInternal; i++)
    {
        const int j = internalNodes[i];
        result[j] = qInt[i];
    }
    for (const auto [j, value] : dirichletNodes)
    {
        result[j] = value;
    }

    return result;
}

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./HeatEquation <msh file>";
    if (argc != 2)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string meshFileName = argv[1];

    auto triMesh = mesh::parseTriangleGmsh(meshFileName);

    const auto elementType = el::Type::P1;
    const auto baseElement = el::createElement(elementType);

    auto mesh = mesh::createMesh(triMesh, baseElement);

    std::vector<float> nodeValues = solveHeatEquation4(mesh);
    // std::vector<float> nodeValues = solveHeatEquation2(mesh);
    // std::vector<float> nodeValues = solveHeatEquation3(mesh);

    float minVal = std::numeric_limits<float>::infinity();
    float maxVal = -std::numeric_limits<float>::infinity();
    for (const auto v : nodeValues)
    {
        minVal = std::min(minVal, v);
        maxVal = std::max(maxVal, v);
    }
    std::cout << "Min = " << minVal << ", max = " << maxVal << "\n";

    mesh::Interpolator interp(mesh, 0.05);
    mesh::SimpleColorScale scc(minVal, maxVal,
                               {cv::Scalar(128, 0, 0), cv::Scalar(0, 0, 128), cv::Scalar(0, 255, 255)});

    interp.setValues(nodeValues);
    cv::Mat outImg = mesh::drawValues(interp, scc, 1000);
    cv::imwrite("values_heatEquation.png", outImg);

    if (true)
    {
        const cv::Mat img = mesh::drawMesh(mesh, 3500);
        cv::imwrite("mesh.png", img);
    }

    return 0;
}
