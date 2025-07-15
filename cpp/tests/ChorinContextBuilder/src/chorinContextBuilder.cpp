#include <format>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include <mesh/concreteMesh.h>
#include <mesh/io.h>

#include <element/factory.h>

#include <fem/chorinCsr.h>
#include <fem/fastConvection.h>

#include <linalg/csrMatrix.h>
#include <linalg/eigen.h>

#include <utils/stopwatch.h>

using SpMat = linalg::CsrMatrix<float>;

template <typename F>
size_t vecSize(const std::vector<F> & v)
{
    return v.size() * sizeof(F);
}

template <typename F>
size_t csrSize(const linalg::CsrMatrix<F> & m)
{
    size_t sum = vecSize(m.values);
    sum += vecSize(m.column);
    sum += vecSize(m.rowStart);
    return sum;
}

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./ChorinContextBuilder <mesh file>";
    if (argc != 2)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string meshFile = argv[1];

    const auto triMesh = mesh::parseTriangleGmsh(meshFile);
    std::cout << std::format("Nodes = {}, elements = {}\n", triMesh.nodes.size(), triMesh.elements.size());

    const auto velocityElement = el::createElement(el::Type::P2);
    const auto pressureElement = el::createElement(el::Type::P1);
    const int integrationDegree = 4;

    const auto velocityMesh = mesh::createMesh(triMesh, *velocityElement);
    const auto pressureMesh = mesh::createMesh(triMesh, *pressureElement);

    const int nThreads = 8;
    auto matrices = fem::buildChorinCsrMatrices<float>(velocityMesh, pressureMesh, integrationDegree, nThreads);

    el::TriangleIntegrator velocityIntegrator(velocityMesh.baseElement, integrationDegree, pressureMesh.baseElement);
    fem::FastConvection fastConvection(velocityMesh, velocityIntegrator);
    auto convection = linalg::csrFromEigen(fastConvection.convection);
    auto fastConvectionIntegration = linalg::csrFromEigen(fastConvection.integration);

    const size_t sizeV0 = csrSize(matrices.velocityMass);
    const size_t sizeVC = csrSize(convection);
    const size_t sizeF = csrSize(fastConvectionIntegration);

    std::cout << "num_velocity_nodes,velocity_mass_size,velocity_convection_size,velocity_f_size\n";
    std::cout << velocityMesh.nodes.size() << "," << sizeV0 << "," << sizeVC << "," << sizeF << "\n";

    return 0;
}