#include <format>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include <mesh/concreteMesh.h>
#include <mesh/io.h>

#include <element/factory.h>

#include <fem/chorinCsr.h>
#include <fem/fastConvection.h>
#include <fem/slowConvection.h>
#include <fem/verySlowConvection.h>

#include <linalg/csrMatrix.h>
#include <linalg/eigen.h>
#include <linalg/io.h>

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

void measureBuildTimes(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                       const int integrationDegree, const int maxThreads)
{
    const int numRuns = 16;
    std::vector<std::vector<float>> times(maxThreads);
    for (int i = 0; i < maxThreads; i++)
    {
        const int numThreads = i + 1;
        auto & currT = times[i];
        currT.resize(numRuns);
        for (int j = 0; j < numRuns; j++)
        {
            u::Stopwatch sw;
            auto matrices = fem::buildChorinCsrMatrices<float>(velocityMesh, pressureMesh, integrationDegree, numThreads);
            const float t = sw.millis();
            currT[j] = t;
            std::cout << std::format("Threads [{}] {}/{}: {} ms\n", numThreads, j + 1, numRuns, t);
        }
    }

    std::cout << "run";
    for (int i = 0; i < maxThreads; i++)
    {
        std::cout << "," << i + 1 << "T";
    }
    std::cout << "\n";

    for (int j = 0; j < numRuns; j++)
    {
        std::cout << j + 1;
        for (int i = 0; i < maxThreads; i++)
        {
            std::cout << "," << times[i][j];
        }
        std::cout << "\n";
    }
}

void measureConvectionTimes(const mesh::ConcreteMesh & velocityMesh, const int integrationDegree, const std::vector<float> & velocityXy)
{
    el::TriangleIntegrator integrator(velocityMesh.baseElement, integrationDegree);

    u::Stopwatch setupSw;
    fem::FastConvection fast(velocityMesh, integrator);
    const auto tSetupFast = setupSw.millis(true);
    fem::SlowConvection slow(velocityMesh, integrator);
    const auto tSetupSlow = setupSw.millis(false);
    fem::VerySlowConvection verySlow(velocityMesh, integrator);

    std::cout << std::format("Setup times: fast = {} ms, slow = {} ms\n", tSetupFast, tSetupSlow);
    // return;

    auto & fastConv = fast.convection;
    auto & slowConv = slow.convection;

    const int n = velocityMesh.nodes.size();
    if (velocityXy.size() != 2 * n)
    {
        std::cerr << "VelocityXy vector has a wrong size!\n";
        return;
    }
    if (slowConv.rows() != n || slowConv.cols() != n)
    {
        std::cerr << "Bad slow convection format!\n";
        return;
    }
    if (fastConv.rows() != n || fastConv.cols() != n)
    {
        std::cerr << "Bad fast convection format!\n";
        return;
    }
    const int nnz = fastConv.nonZeros();
    if (nnz != slowConv.nonZeros())
    {
        std::cerr << "Mismatch between nnz of the two matrices!\n";
        return;
    }

    {
        const auto colPtrFast = fastConv.innerIndexPtr();
        const auto colPtrSlow = slowConv.innerIndexPtr();
        for (int i = 0; i < nnz; i++)
        {
            if (colPtrFast[i] != colPtrSlow[i])
            {
                std::cerr << "Mismatch in sprase structure!\n";
                return;
            }
        }
    }

    float * fastVals = fastConv.valuePtr();
    float * slowVals = slowConv.valuePtr();

    const int numRuns = 16;
    std::vector<float> timesFast(numRuns);
    std::vector<float> timesSlow(numRuns);
    std::vector<float> timesVerySlow(numRuns);

    for (int i = 0; i < numRuns; i++)
    {
        u::Stopwatch sw;
        fast.update(velocityXy);
        timesFast[i] = sw.millis(true);
        slow.update(velocityXy);
        timesSlow[i] = sw.millis(true);
        auto verySlowConv = verySlow.calculate(velocityXy);
        timesVerySlow[i] = sw.millis();

        double vsSumSq = 0;
        float vsAbsMax = 0;
        float vsMse = 0;
        if (verySlowConv.rows() != n || verySlowConv.cols() != n || verySlowConv.nonZeros() != nnz)
        {
            std::cerr << "Bad very slow conv!\n";
        }
        else
        {
            const float * verySlowVals = verySlowConv.valuePtr();
            for (int j = 0; j < nnz; j++)
            {
                const float fastVal = fastVals[j];
                const float slowVal = verySlowVals[j];
                const float d = std::abs(fastVal - slowVal);
                vsAbsMax = std::max(vsAbsMax, d);
                vsSumSq += d * d;
            }
            vsMse = std::sqrt(vsSumSq / nnz);
        }

        double sumSq = 0;
        float absMax = 0;
        for (int j = 0; j < nnz; j++)
        {
            const float fastVal = fastVals[j];
            const float slowVal = slowVals[j];
            const float d = std::abs(fastVal - slowVal);
            absMax = std::max(absMax, d);
            sumSq += d * d;
        }
        const float mse = std::sqrt(sumSq / nnz);
        std::cout << std::format("Run {}/{}: fast = {} ms, slow = {} ms, very slow = {} ms, absMax / mse slow = {} / {}, absMax / mse very slow = {} / {}\n",
                                 i + 1, numRuns, timesFast[i], timesSlow[i], timesVerySlow[i],
                                 absMax, mse, vsAbsMax, vsMse);
    }

    std::cout << "run,fast,slow,very_slow\n";
    for (int i = 0; i < numRuns; i++)
    {
        std::cout << i + 1 << "," << timesFast[i] << "," << timesSlow[i] << "," << timesVerySlow[i] << "\n";
    }
}

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./ChorinContextBuilder <mesh file> (nothing | -m | -p <velocityXy>)";
    if (argc < 2)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string meshFile = argv[1];

    const auto triMesh = mesh::parseTriangleGmsh(meshFile);
    std::cout << std::format("Nodes = {}, triangles = {}\n", triMesh.nodes.size(), triMesh.elements.size());

    const auto velocityElement = el::createElement(el::Type::P2);
    const auto pressureElement = el::createElement(el::Type::P1);
    const int integrationDegree = 4;

    const auto velocityMesh = mesh::createMesh(triMesh, *velocityElement);
    const auto pressureMesh = mesh::createMesh(triMesh, *pressureElement);

    std::cout << std::format("Velocity nodes = {}, pressure nodes = {}\n", velocityMesh.nodes.size(), pressureMesh.nodes.size());
    if (argc > 2 && std::string(argv[2]) == "-m")
    {
        std::cout << "Measuring matrix assembly times\n";
        measureBuildTimes(velocityMesh, pressureMesh, integrationDegree, 8);
        return 0;
    }
    if (argc == 4 && std::string(argv[2]) == "-p")
    {
        std::cout << "Measure convection assembly times\n";
        auto velocityXy = linalg::readVec<float>(argv[3]);
        measureConvectionTimes(velocityMesh, integrationDegree, velocityXy);
        return 0;
    }

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