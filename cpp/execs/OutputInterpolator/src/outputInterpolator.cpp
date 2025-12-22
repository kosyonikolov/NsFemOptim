#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <fstream>

#include <mesh/colorScale.h>
#include <mesh/drawMesh.h>
#include <mesh/io.h>
#include <mesh/triangleLookup.h>

#include <element/factory.h>

#include <linalg/io.h>

#include <OutputInterpolator/config.h>

struct Triplet
{
    int id;
    std::string velocityFname;
    std::string pressureFname;
};

struct PressureRange
{
    float min;
    float max;
};

std::vector<Triplet> findFiles(const std::string & dir)
{
    struct Pair
    {
        int id;
        std::string fileName;

        bool operator<(const Pair & other) const
        {
            return id < other.id;
        }
    };

    std::vector<Pair> velocityPairs;
    std::vector<Pair> pressurePairs;

    for (const auto & entry : std::filesystem::directory_iterator(dir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        if (entry.path().extension() != ".bin")
        {
            continue;
        }

        // Match pressure_<id> and velocity_<id>
        const std::string stem = entry.path().stem().generic_string();
        const auto sepIdx = stem.find('_');
        if (sepIdx == stem.npos)
        {
            continue;
        }

        const std::string type = stem.substr(0, sepIdx);
        const std::string idStr = stem.substr(sepIdx + 1);
        int id = 0;
        std::istringstream iss(idStr);
        if (!(iss >> id))
        {
            continue;
        }

        const std::string fileName = entry.path().generic_string();
        if (type == "pressure")
        {
            pressurePairs.emplace_back(id, fileName);
        }
        else
        {
            velocityPairs.emplace_back(id, fileName);
        }
    }

    std::sort(velocityPairs.begin(), velocityPairs.end());
    std::sort(pressurePairs.begin(), pressurePairs.end());

    std::vector<Triplet> result;
    // Merge sort ftw
    const size_t nV = velocityPairs.size();
    const size_t nP = pressurePairs.size();
    size_t iV = 0;
    size_t iP = 0;
    while (iV < nV && iP < nP)
    {
        const auto & vp = velocityPairs[iV];
        const auto & pp = pressurePairs[iP];
        if (vp.id == pp.id)
        {
            result.emplace_back(vp.id, vp.fileName, pp.fileName);
            iV++;
            iP++;
        }
        else if (vp.id > pp.id)
        {
            std::cerr << std::format("No velocity file for pressure ID {}!\n", pp.id);
            iP++;
        }
        else // pp.id > vp.id
        {
            std::cerr << std::format("No pressure file for velocity ID {}!\n", vp.id);
            iV++;
        }
    }

    while (iV < nV)
    {
        std::cerr << std::format("No pressure file for velocity ID {}!\n", velocityPairs[iV].id);
        iV++;
    }

    while (iP < nP)
    {
        std::cerr << std::format("No velocity file for pressure ID {}!\n", pressurePairs[iP].id);
        iP++;
    }

    return result;
}

void writePressureStats(const std::vector<Triplet> & triplets, const std::string & outFileName)
{
    if (triplets.empty())
    {
        std::cerr << "!!! No triplets !!!\n";
        return;
    }

    std::ofstream file(outFileName);
    file << "id,min,max\n";

    const size_t n = triplets.size();
    for (size_t i = 0; i < n; i++)
    {
        auto curr = linalg::readVec<float>(triplets[i].pressureFname);
        auto minMax = std::minmax_element(curr.begin(), curr.end());
        file << std::format("{},{},{}\n", i, *minMax.first, *minMax.second);
    }
}

PressureRange findPressureRange(const std::vector<Triplet> & triplets, int skipSteps)
{
    if (triplets.empty())
    {
        std::cerr << "!!! No triplets !!!\n";
        PressureRange result;
        result.min = 0;
        result.max = 1;
        return result;
    }

    const size_t n = triplets.size();
    if (skipSteps >= n)
    {
        std::cerr << "!!! Pressure skip steps are more than the number of triplets, ignoring them !!!";
        skipSteps = 0;
    }

    PressureRange result;
    result.min = std::numeric_limits<float>::max();
    result.max = std::numeric_limits<float>::min();
    for (size_t i = skipSteps; i < n; i++)
    {
        auto curr = linalg::readVec<float>(triplets[i].pressureFname);
        auto minMax = std::minmax_element(curr.begin(), curr.end());
        result.min = std::min(result.min, *minMax.first);
        result.max = std::max(result.max, *minMax.second);
    }

    result.max += 1e-3f;

    return result;
}

class Renderer
{
    std::string outputDir;
    const Config & cfg;
    const mesh::ConcreteMesh & velocityMesh;
    const mesh::ConcreteMesh & pressureMesh;
    const mesh::SimpleColorScale & pressureColorScale;

    mesh::TriangleLookup lookup;

public:
    Renderer(const std::string & outputDir, const Config & cfg,
             const mesh::ConcreteMesh & velocityMesh,
             const mesh::ConcreteMesh & pressureMesh,
             const mesh::SimpleColorScale & pressureColorScale)
        : outputDir(outputDir), cfg(cfg), velocityMesh(velocityMesh),
          pressureMesh(pressureMesh), pressureColorScale(pressureColorScale),
          lookup(velocityMesh, 0.05)
    {
    }

    void render(const Triplet & t)
    {
        const auto nV = 2 * velocityMesh.nodes.size();
        const auto nP = pressureMesh.nodes.size();

        const std::string outFname = std::format("{}/out_{}.{}", outputDir, t.id, cfg.ext);

        auto velocity = linalg::readVec<float>(t.velocityFname);
        if (velocity.size() != nV)
        {
            std::cerr << "!!! SKIP " + outFname + " - velocity file has wrong size !!!\n";
            return;
        }

        auto pressure = linalg::readVec<float>(t.pressureFname);
        if (pressure.size() != nP)
        {
            std::cerr << "!!! SKIP " + outFname + " - pressure file has wrong size !!!\n";
            return;
        }

        const cv::Mat img = mesh::drawCfd(lookup, pressureColorScale, 800,
                                          cfg.velocityScale, cfg.velocityStep,
                                          velocityMesh, pressureMesh,
                                          velocity, pressure);
        std::cout << outFname + "\n";
        cv::imwrite(outFname, img);
    }
};

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./OutputInterpolator <cfg> <msh file> <output dir>";
    if (argc != 4)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string cfgFname = argv[1];
    const std::string meshFileName = argv[2];
    const std::string outputDir = argv[3];

    auto cfg = parseConfig(cfgFname);

    std::cout << "Parsing mesh... ";
    std::cout.flush();
    auto triMesh = mesh::parseTriangleGmsh(meshFileName);
    std::cout << "Done\n";

    const auto velocityElement = el::createElement(el::Type::P2);
    const auto pressureElement = el::createElement(el::Type::P1);

    std::cout << "Creating pressure and velocity meshes... ";
    std::cout.flush();
    auto velocityMesh = mesh::createMesh(triMesh, *velocityElement);
    auto pressureMesh = mesh::createMesh(triMesh, *pressureElement);
    std::cout << "Done\n";

    const auto triplets = findFiles(outputDir);
    std::cout << "Found " << triplets.size() << " output files\n";

    if (cfg.dumpPressureStats)
    {
        const std::string statsFileName = "pressure_stats.csv";
        std::cout << "Saving pressure stats to " << statsFileName << "\n";
        writePressureStats(triplets, statsFileName);
    }

    PressureRange pressureRange;
    if (cfg.manualPressure)
    {
        pressureRange.min = cfg.minPressure;
        pressureRange.max = cfg.maxPressure;
    }
    else
    {
        pressureRange = findPressureRange(triplets, cfg.pressureSkipSteps);
    }

    std::cout << std::format("Pressure range = [{}, {}]\n", pressureRange.min, pressureRange.max);

    std::vector<cv::Scalar> colors{cv::Scalar(128, 0, 0), cv::Scalar(0, 0, 128), cv::Scalar(0, 200, 200)};
    mesh::SimpleColorScale pressureColorScale(pressureRange.min, pressureRange.max, colors);

    std::atomic<size_t> i = 0;
    const size_t nT = triplets.size();

    auto renderFn = [&]()
    {
        Renderer renderer(outputDir, cfg, velocityMesh, pressureMesh, pressureColorScale);

        while (true)
        {
            size_t idx = i.fetch_add(1);
            if (idx >= nT)
            {
                break;
            }

            const auto triplet = triplets[idx];
            try
            {
                renderer.render(triplet);
            }
            catch (std::exception & e)
            {
                std::cerr << "!!! Failed to render triplet !!!\n";
                break;
            }
        }
    };

    const int nThreads = cfg.numThreads;
    if (nThreads < 1 || nThreads > 32)
    {
        std::cerr << "Bad number of threads! Valid values are in [1, 32]\n";
        return 1;
    }

    std::vector<std::thread> threads(nThreads);
    for (int i = 0; i < nThreads; i++)
    {
        threads[i] = std::thread(renderFn);
    }

    for (int i = 0; i < nThreads; i++)
    {
        threads[i].join();
    }

    return 0;
}