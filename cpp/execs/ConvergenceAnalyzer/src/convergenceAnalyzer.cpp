#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

#include <linalg/io.h>

#include <ConvergenceAnalyzer/config.h>

enum class MapType
{
    Identity,
    Standard,
    Undefined
};

struct InputConfig
{
    std::string smallDir;
    std::string mediumDir, mediumVelocityMap, mediumPressureMap;
    std::string largeDir, largeVelocityMap, largePressureMap;
};

struct Triplet
{
    int id;
    std::string velocityFname;
    std::string pressureFname;
};

struct Stats
{
    std::vector<int> id;
    // Differences per timestep
    std::vector<float> smallToMedium, mediumToLarge;
    // Convergence rate per timestep
    std::vector<float> convRate;
    float globalConvRate;
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

InputConfig readInputConfig(const std::string & fileName)
{
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        throw std::runtime_error(std::format("Failed to open input config file [{}]", fileName));
    }

    InputConfig result;
    if (!(file >> result.smallDir >> result.mediumDir >> result.mediumVelocityMap >> result.mediumPressureMap >>
          result.largeDir >> result.largeVelocityMap >> result.largePressureMap))
    {
        throw std::runtime_error("Failed to parse input config");
    }
    return result;
}

std::vector<int> readMap(const std::string & fileName)
{
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        throw std::runtime_error(std::format("Failed to open map file [{}]", fileName));
    }

    std::vector<int> result;
    int id;
    while (file >> id)
    {
        result.push_back(id);
    }
    return result;
}

std::vector<std::vector<int>> createCommonMaps(const std::vector<std::vector<int>> & maps)
{
    if (maps.empty())
    {
        throw std::invalid_argument("No input maps!");
    }
    const int m = maps.size();
    std::vector<std::vector<int>> result(m + 1);

    int n = maps[0].size();
    for (int i = 1; i < m; i++)
    {
        n = std::min<int>(n, maps[i].size());
    }

    // Only include nodes which are present in all maps
    std::vector<int> curr(m);
    for (int i = 0; i < n; i++)
    {
        bool ok = true;
        for (int j = 0; j < m; j++)
        {
            curr[j] = maps[j][i];
            if (curr[j] < 0)
            {
                ok = false;
                break;
            }
        }

        if (!ok)
        {
            std::cerr << std::format("Skip {} - not present in all maps\n", i);
            continue;
        }

        result[0].push_back(i); // Small mesh
        for (int j = 0; j < m; j++)
        {
            result[j + 1].push_back(curr[j]);
        }
    }

    return result;
}

Stats analyze(const std::vector<std::vector<Triplet>> & dirs, const std::vector<std::vector<int>> & maps, const bool velocity)
{
    const int m = 3;
    if (dirs.size() != m || maps.size() != m)
    {
        throw std::invalid_argument("There should be exactly 3 directories and maps");
    }

    const int nT = dirs[0].size();
    // Sanity check
    for (int i = 1; i < m; i++)
    {
        if (dirs[i].size() != nT)
        {
            throw std::invalid_argument(std::format("Directory [{}] has wrong number of timesteps!\n", i));
        }
        for (int j = 0; j < nT; j++)
        {
            if (dirs[i][j].id != dirs[0][j].id)
            {
                throw std::invalid_argument(std::format("Different timesteps detected in directory [{}]!\n", i));
            }
        }
    }

    const int nNodes = maps[0].size();
    for (int i = 1; i < m; i++)
    {
        if (maps[i].size() != nNodes)
        {
            throw std::invalid_argument(std::format("Map [{}] has wrong size!\n", i));
        }
    }

    Stats result;
    double globalS2MSum = 0;
    double globalM2LSum = 0;

    // Only common nodes
    std::vector<std::vector<float>> values(m);
    const int nTotalNodes = velocity ? 2 * nNodes : nNodes;
    for (int i = 0; i < m; i++)
    {
        values[i].resize(nTotalNodes);
    }

    for (int i = 0; i < nT; i++)
    {
        result.id.push_back(dirs[0][i].id);
        // Load common nodes
        for (int j = 0; j < m; j++)
        {
            const auto & triplet = dirs[j][i];
            const std::string fileName = velocity ? triplet.velocityFname : triplet.pressureFname;
            auto curr = linalg::readVec<float>(fileName);
            const int lim = curr.size();
            const auto & theMap = maps[j];
            for (int k = 0; k < nNodes; k++)
            {
                const int idx = theMap[k];
                if (idx < 0 || idx >= lim)
                {
                    throw std::invalid_argument("Map contains invalid index");
                }
                values[j][k] = curr[idx];
                if (velocity)
                {
                    const int yIdx = idx + curr.size() / 2;
                    values[j][k + nNodes] = curr[yIdx];
                }
            }
        }

        // Calculate differences
        double s2mSum = 0;
        double m2lSum = 0;
        for (int j = 0; j < nTotalNodes; j++)
        {
            const double dM = values[0][j] - values[1][j];
            const double dL = values[1][j] - values[2][j];
            s2mSum += dM * dM;
            m2lSum += dL * dL;
        }

        globalS2MSum += s2mSum;
        globalM2LSum += m2lSum;

        // Calculate local MSE and convergence
        const double s2m = std::sqrt(s2mSum / nTotalNodes);
        const double m2l = std::sqrt(m2lSum / nTotalNodes);
        const double ratio = m2l / s2m;
        const double currRate = std::log(ratio) / std::log(0.5);

        result.smallToMedium.push_back(s2m);
        result.mediumToLarge.push_back(m2l);
        result.convRate.push_back(currRate);
    }

    // Global stats
    const double denom = 1.0 / (static_cast<int64_t>(nT) * nTotalNodes);
    const double s2m = std::sqrt(globalS2MSum * denom);
    const double m2l = std::sqrt(globalM2LSum * denom);
    const double ratio = m2l / s2m;
    result.globalConvRate = std::log(ratio) / std::log(0.5);

    return result;
}

void writeStats(const Stats & stats, const std::string & fileName)
{
    std::ofstream file(fileName);
    file << "id,s2m,m2l,conv_rate\n";
    const int n = stats.id.size();
    for (int i = 0; i < n; i++)
    {
        file << std::format("{},{},{},{}\n", stats.id[i], stats.smallToMedium[i], stats.mediumToLarge[i], stats.convRate[i]);
    }
    file << "\n";
    file << "global_conv_rate," << stats.globalConvRate << "\n";
}

MapType determineMapType(const InputConfig & cfg)
{
    const bool a = cfg.mediumVelocityMap != "-";
    const bool b = cfg.largeVelocityMap != "-";
    const bool c = cfg.mediumPressureMap != "-";
    const bool d = cfg.largePressureMap != "-";

    if (a && b && c && d)
    {
        return MapType::Standard;
    }
    else if (!a && !b && !c && !d)
    {
        return MapType::Identity;
    }
    return MapType::Undefined;
}

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./ConvergenceAnalyzer <cfg> <input config>";
    if (argc != 3)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string cfgFname = argv[1];
    const std::string inputCfgFname = argv[2];

    auto cfg = parseConfig(cfgFname);
    const auto inputCfg = readInputConfig(inputCfgFname);

    std::vector<std::vector<Triplet>> dirs(3);
    dirs[0] = findFiles(inputCfg.smallDir);
    dirs[1] = findFiles(inputCfg.mediumDir);
    dirs[2] = findFiles(inputCfg.largeDir);

    if (cfg.replaceInputIds)
    {
        std::cout << "Replacing input IDs!!!\n";
        for (auto & dir : dirs)
        {
            int i = 0;
            for (auto & t : dir)
            {
                t.id = i;
                i++;
            }
        }
    }

    if (dirs[0].empty())
    {
        std::cerr << "No files found in small directory!\n";
        return 1;
    }

    std::vector<std::vector<int>> velocityMaps, pressureMaps;

    const auto mapType = determineMapType(inputCfg);
    if (mapType == MapType::Standard)
    {
        std::cout << "Reading maps for spatial convergence\n";
        std::vector<std::vector<int>> inputVelocityMaps(2), inputPressureMaps(2);
        inputVelocityMaps[0] = readMap(inputCfg.mediumVelocityMap);
        inputVelocityMaps[1] = readMap(inputCfg.largeVelocityMap);
        inputPressureMaps[0] = readMap(inputCfg.mediumPressureMap);
        inputPressureMaps[1] = readMap(inputCfg.largePressureMap);

        velocityMaps = createCommonMaps(inputVelocityMaps);
        pressureMaps = createCommonMaps(inputPressureMaps);
    }
    else if (mapType == MapType::Identity)
    {
        std::cout << "Creating identity maps for temporal convergence\n";

        // Read one vector to determine the number of velocity and pressure nodes
        const auto velocity = linalg::readVec<float>(dirs[0][0].velocityFname);
        const auto pressure = linalg::readVec<float>(dirs[0][0].pressureFname);
        const int nVelocityNodes = velocity.size() / 2; // !!! 2 channels (X and Y) !!!
        const int nPressureNodes = pressure.size();
        std::cout << std::format("Velocity nodes = {}, pressure nodes = {}\n", nVelocityNodes, nPressureNodes);

        std::vector<int> vMap(nVelocityNodes);
        std::iota(vMap.begin(), vMap.end(), 0);

        std::vector<int> pMap(nPressureNodes);
        std::iota(pMap.begin(), pMap.end(), 0);

        for (int i = 0; i < 3; i++)
        {
            velocityMaps.push_back(vMap);
            pressureMaps.push_back(pMap);
        }
    }
    else
    {
        std::cerr << "Couldn't determine map type! Either all maps should be disabled (-), or all should be present!\n";
        return 1;
    }

    const auto pressureStats = analyze(dirs, pressureMaps, false);
    const auto velocityStats = analyze(dirs, velocityMaps, true);

    writeStats(pressureStats, "pressure_stats.csv");
    writeStats(velocityStats, "velocity_stats.csv");

    std::cout << std::format("Velocity convergence rate: {}\nPressure convergence rate: {}\n", velocityStats.globalConvRate, pressureStats.globalConvRate);

    return 0;
}