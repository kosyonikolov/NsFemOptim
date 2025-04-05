#include <fem/chorinCsr.h>

#include <atomic>
#include <cassert>
#include <future>

#include <linalg/sparseDokBuilder.h>

#include <element/triangleIntegrator.h>

#include <utils/stopwatch.h>

namespace fem
{
    template <typename F>
    struct PrototypeBundle
    {
        linalg::CsrMatrix<F> velocity;
        linalg::CsrMatrix<F> pressure;
        linalg::CsrMatrix<F> velocityPressureDiv;
        linalg::CsrMatrix<F> pressureVelocityDiv;
    };

    template <typename F>
    struct ValueBundle
    {
    private:
        void addVecs(std::vector<F> & dst, const std::vector<F> & src)
        {
            if (dst.size() != src.size())
            {
                throw std::invalid_argument(std::format("{}: Mismatching vector sizes", __FUNCTION__));
            }
            const size_t n = src.size();
            for (size_t i = 0; i < n; i++)
            {
                dst[i] += src[i];
            }
        }

    public:
        std::vector<F> velocityMass;
        std::vector<F> velocityStiffness;
        std::vector<F> pressureStiffness;
        std::vector<F> velocityPressureDiv;
        std::vector<F> pressureVelocityDiv;

        ValueBundle<F> & operator+=(const ValueBundle<F> & other)
        {
            addVecs(velocityMass, other.velocityMass);
            addVecs(velocityStiffness, other.velocityStiffness);
            addVecs(pressureStiffness, other.pressureStiffness);
            addVecs(velocityPressureDiv, other.velocityPressureDiv);
            addVecs(pressureVelocityDiv, other.pressureVelocityDiv);

            return *this;
        }
    };

    template <typename F>
    PrototypeBundle<F> buildPrototypes(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh)
    {
        if (velocityMesh.numElements != pressureMesh.numElements)
        {
            throw std::invalid_argument(std::format("{}: Velocity and pressure meshes have different number of elements", __FUNCTION__));
        }

        const int numVelocityNodes = velocityMesh.nodes.size(); // ! Counts only one channel, not X + Y - multiply by 2 to get all velocity nodes
        const int numPressureNodes = pressureMesh.nodes.size();
        const int elSizeV = velocityMesh.getElementSize();
        const int elSizeP = pressureMesh.getElementSize();

        // Build prototype matrices first - structure only
        using Builder = linalg::SparseMatrixPrototypeBuilder;

        Builder velocityProtoBuilder(numVelocityNodes, numVelocityNodes);
        Builder pressureProtoBuilder(numPressureNodes, numPressureNodes);
        Builder velocityPressureDivProtoBuilder(numPressureNodes, 2 * numVelocityNodes);
        Builder pressureVelocityDivProtoBuilder(2 * numVelocityNodes, numPressureNodes);

        std::vector<int> idsV(elSizeV);
        std::vector<int> idsP(elSizeP);

        // u::Stopwatch sw;
        const int nElem = velocityMesh.numElements;
        for (int i = 0; i < nElem; i++)
        {
            // ========================= Velocity-only matrices =========================
            velocityMesh.getElement(i, idsV.data(), 0);
            for (int r = 0; r < elSizeV; r++)
            {
                const int i = idsV[r];
                for (int c = 0; c < elSizeV; c++)
                {
                    const int j = idsV[c];
                    velocityProtoBuilder.add(i, j);
                }
            }

            // ========================= Pressure-only matrices =========================
            pressureMesh.getElement(i, idsP.data(), 0);
            for (int r = 0; r < elSizeP; r++)
            {
                const int i = idsP[r];
                for (int c = 0; c < elSizeP; c++)
                {
                    const int j = idsP[c];
                    pressureProtoBuilder.add(i, j);
                }
            }

            // ========================= Divergence matrices =========================
            for (int iV = 0; iV < elSizeV; iV++)
            {
                const int gV = idsV[iV];
                for (int iP = 0; iP < elSizeP; iP++)
                {
                    const int gP = idsP[iP];
                    // clang-format off
                    velocityPressureDivProtoBuilder.add(gP, gV                   );
                    velocityPressureDivProtoBuilder.add(gP, gV + numVelocityNodes);
                    pressureVelocityDivProtoBuilder.add(gV, gP                   );
                    pressureVelocityDivProtoBuilder.add(gV + numVelocityNodes, gP);
                    // clang-format on
                }
            }
        }
        // const auto tElements = sw.millis(true);

        PrototypeBundle<F> result;
        result.velocity = velocityProtoBuilder.buildCsrPrototype2<F>();
        result.pressure = pressureProtoBuilder.buildCsrPrototype2<F>();
        result.velocityPressureDiv = velocityPressureDivProtoBuilder.buildCsrPrototype2<F>();
        result.pressureVelocityDiv = pressureVelocityDivProtoBuilder.buildCsrPrototype2<F>();
        // const auto tAsm = sw.millis();
        // std::cout << std::format("{}: elements = {}, assemble = {}\n", __FUNCTION__, tElements, tAsm);

        return result;
    }

    template <typename F>
    PrototypeBundle<F> buildPrototypes4T(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh)
    {
        if (velocityMesh.numElements != pressureMesh.numElements)
        {
            throw std::invalid_argument(std::format("{}: Velocity and pressure meshes have different number of elements", __FUNCTION__));
        }

        const int numVelocityNodes = velocityMesh.nodes.size(); // ! Counts only one channel, not X + Y - multiply by 2 to get all velocity nodes
        const int numPressureNodes = pressureMesh.nodes.size();
        const int elSizeV = velocityMesh.getElementSize();
        const int elSizeP = pressureMesh.getElementSize();

        // Build prototype matrices first - structure only
        using Builder = linalg::SparseMatrixPrototypeBuilder;

        Builder velocityProtoBuilder(numVelocityNodes, numVelocityNodes);
        Builder pressureProtoBuilder(numPressureNodes, numPressureNodes);
        Builder velocityPressureDivProtoBuilder(numPressureNodes, 2 * numVelocityNodes);
        Builder pressureVelocityDivProtoBuilder(2 * numVelocityNodes, numPressureNodes);

        const int nElem = velocityMesh.numElements;
        PrototypeBundle<F> result;

        auto buildVelocity = [&]() -> void
        {
            std::vector<int> idsV(elSizeV);
            for (int i = 0; i < nElem; i++)
            {
                velocityMesh.getElement(i, idsV.data(), 0);
                for (int r = 0; r < elSizeV; r++)
                {
                    const int i = idsV[r];
                    for (int c = 0; c < elSizeV; c++)
                    {
                        const int j = idsV[c];
                        velocityProtoBuilder.add(i, j);
                    }
                }
            }
            result.velocity = velocityProtoBuilder.buildCsrPrototype2<F>();
        };

        auto buildPressure = [&]() -> void
        {
            std::vector<int> idsP(elSizeP);
            for (int i = 0; i < nElem; i++)
            {
                pressureMesh.getElement(i, idsP.data(), 0);
                for (int r = 0; r < elSizeP; r++)
                {
                    const int i = idsP[r];
                    for (int c = 0; c < elSizeP; c++)
                    {
                        const int j = idsP[c];
                        pressureProtoBuilder.add(i, j);
                    }
                }
            }
            result.pressure = pressureProtoBuilder.buildCsrPrototype2<F>();
        };

        auto buildVpd = [&]() -> void
        {
            std::vector<int> idsV(elSizeV);
            std::vector<int> idsP(elSizeP);
            for (int i = 0; i < nElem; i++)
            {
                velocityMesh.getElement(i, idsV.data(), 0);
                pressureMesh.getElement(i, idsP.data(), 0);
                for (int iV = 0; iV < elSizeV; iV++)
                {
                    const int gV = idsV[iV];
                    for (int iP = 0; iP < elSizeP; iP++)
                    {
                        const int gP = idsP[iP];
                        // clang-format off
                        velocityPressureDivProtoBuilder.add(gP, gV                   );
                        velocityPressureDivProtoBuilder.add(gP, gV + numVelocityNodes);
                        // clang-format on
                    }
                }
            }
            result.velocityPressureDiv = velocityPressureDivProtoBuilder.buildCsrPrototype2<F>();
        };

        auto buildPvd = [&]() -> void
        {
            std::vector<int> idsV(elSizeV);
            std::vector<int> idsP(elSizeP);
            for (int i = 0; i < nElem; i++)
            {
                velocityMesh.getElement(i, idsV.data(), 0);
                pressureMesh.getElement(i, idsP.data(), 0);
                for (int iV = 0; iV < elSizeV; iV++)
                {
                    const int gV = idsV[iV];
                    for (int iP = 0; iP < elSizeP; iP++)
                    {
                        const int gP = idsP[iP];
                        // clang-format off
                        pressureVelocityDivProtoBuilder.add(gV, gP                   );
                        pressureVelocityDivProtoBuilder.add(gV + numVelocityNodes, gP);
                        // clang-format on
                    }
                }
            }
            result.pressureVelocityDiv = pressureVelocityDivProtoBuilder.buildCsrPrototype2<F>();
        };

        std::future<void> velocityF = std::async(std::launch::async, buildVelocity);
        std::future<void> pressureF = std::async(std::launch::async, buildPressure);
        std::future<void> vpdF = std::async(std::launch::async, buildVpd);
        buildPvd();

        velocityF.get();
        pressureF.get();
        vpdF.get();

        return result;
    }

    template <typename F>
    void buildIndexLut(const linalg::CsrMatrix<F> & proto, cv::Mat & lut,
                       const std::vector<int> & rowIds, const std::vector<int> & colIds)
    {
        const int nRows = rowIds.size();
        const int nCols = colIds.size();
        lut.create(nRows, nCols, CV_32SC1);
        for (int r = 0; r < nRows; r++)
        {
            const int i = rowIds[r];
            std::span<int> line(lut.ptr<int>(r), nCols);
            proto.findOffsetsUnsorted(i, colIds, line);
        }

        const bool check = true;
        if (check)
        {
            for (int r = 0; r < nRows; r++)
            {
                const int * line = lut.ptr<int>(r);
                for (int c = 0; c < nCols; c++)
                {
                    if (line[c] < 0)
                    {
                        throw std::runtime_error(std::format("{}: Index lookup failed for global index ({}, {})", __FUNCTION__,
                                                             rowIds[r], colIds[c]));
                    }
                }
            }
        }
    }

    template <typename F>
    void buildPart(const PrototypeBundle<F> & proto, ValueBundle<F> & value,
                   const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                   const int integrationDegree, std::atomic<int> & counter)
    {
        const int numVelocityNodes = velocityMesh.nodes.size(); // ! Counts only one channel, not X + Y - multiply by 2 to get all velocity nodes
        // const int numPressureNodes = pressureMesh.nodes.size();
        const int elSizeV = velocityMesh.getElementSize();
        const int elSizeP = pressureMesh.getElementSize();
        const int nElem = velocityMesh.numElements;
        assert(nElem == pressureMesh.numElements);

        el::TriangleIntegrator velocityIntegrator(velocityMesh.baseElement, integrationDegree, pressureMesh.baseElement);
        el::TriangleIntegrator pressureIntegrator(pressureMesh.baseElement, integrationDegree);

        std::vector<int> idsV(elSizeV);
        std::vector<int> idsV2(2 * elSizeV); // idsV with X and Y components
        std::vector<el::Point> ptsV(elSizeV);

        std::vector<int> idsP(elSizeP);
        std::vector<el::Point> ptsP(elSizeP);

        cv::Mat localMassMatrixV, localStiffnessMatrixV;
        cv::Mat localMassMatrixP, localStiffnessMatrixP;
        cv::Mat localVpdX, localVpdY, localPvdX, localPvdY;

        // Index lookups - int32
        cv::Mat velocityLut;
        cv::Mat pressureLut;
        cv::Mat vpdLut;
        cv::Mat pvdLut;

        while (true)
        {
            const int i = counter.fetch_add(1);
            if (i >= nElem)
            {
                break;
            }
            const auto t = velocityMesh.elementTransforms[i];

            // ========================= Velocity-only matrices =========================
            velocityMesh.getElement(i, idsV.data(), ptsV.data());

            velocityIntegrator.integrateLocalMassMatrix(t, localMassMatrixV);
            assert(elSizeV == localMassMatrixV.cols);
            assert(elSizeV == localMassMatrixV.rows);

            velocityIntegrator.integrateLocalStiffnessMatrix(t, localStiffnessMatrixV);
            assert(elSizeV == localStiffnessMatrixV.cols);
            assert(elSizeV == localStiffnessMatrixV.rows);

            buildIndexLut(proto.velocity, velocityLut, idsV, idsV);

            // Accumulate
            for (int r = 0; r < elSizeV; r++)
            {
                const int * lutLine = velocityLut.ptr<int>(r);
                for (int c = 0; c < elSizeV; c++)
                {
                    const int i = lutLine[c];
                    value.velocityMass[i] += localMassMatrixV.at<float>(r, c);
                    value.velocityStiffness[i] += localStiffnessMatrixV.at<float>(r, c);
                }
            }

            // ========================= Pressure-only matrices =========================
            pressureMesh.getElement(i, idsP.data(), ptsP.data());

            pressureIntegrator.integrateLocalStiffnessMatrix(t, localStiffnessMatrixP);
            assert(elSizeP == localStiffnessMatrixP.cols);
            assert(elSizeP == localStiffnessMatrixP.rows);

            buildIndexLut(proto.pressure, pressureLut, idsP, idsP);

            // Accumulate
            for (int r = 0; r < elSizeP; r++)
            {
                const int * lutLine = pressureLut.ptr<int>(r);
                for (int c = 0; c < elSizeP; c++)
                {
                    const int i = lutLine[c];
                    value.pressureStiffness[i] += localStiffnessMatrixP.at<float>(r, c);
                }
            }

            // ========================= Divergence matrices =========================
            velocityIntegrator.integrateLocalDivergenceMatrix(t, false, localVpdX, localVpdY);
            assert(localVpdX.cols == localVpdY.cols && localVpdX.rows == localVpdY.rows);
            assert(localVpdX.rows == elSizeP);
            assert(localVpdX.cols == elSizeV);

            velocityIntegrator.integrateLocalDivergenceMatrix(t, true, localPvdX, localPvdY);
            assert(localPvdX.cols == localPvdY.cols && localPvdX.rows == localPvdY.rows);
            assert(localPvdX.rows == elSizeV);
            assert(localPvdX.cols == elSizeP);

            for (int i = 0; i < elSizeV; i++)
            {
                idsV2[i] = idsV[i];
                idsV2[i + elSizeV] = idsV[i] + numVelocityNodes;
            }

            buildIndexLut(proto.velocityPressureDiv, vpdLut, idsP, idsV2);
            buildIndexLut(proto.pressureVelocityDiv, pvdLut, idsV2, idsP);

            for (int iV = 0; iV < elSizeV; iV++)
            {
                for (int iP = 0; iP < elSizeP; iP++)
                {
                    const int iVpdX = vpdLut.at<int>(iP, iV);
                    const int iVpdY = vpdLut.at<int>(iP, iV + elSizeV);
                    const int iPvdX = pvdLut.at<int>(iV, iP);
                    const int iPvdY = pvdLut.at<int>(iV + elSizeV, iP);

                    // clang-format off
                    value.velocityPressureDiv[iVpdX] += localVpdX.at<float>(iP, iV);
                    value.velocityPressureDiv[iVpdY] += localVpdY.at<float>(iP, iV);
                    value.pressureVelocityDiv[iPvdX] += localPvdX.at<float>(iV, iP);
                    value.pressureVelocityDiv[iPvdY] += localPvdY.at<float>(iV, iP);
                    // clang-format on
                }
            }
        }
    }

    template <typename F>
    void assembleCsrPrototypeValue(linalg::CsrMatrix<F> & csr, const linalg::CsrMatrix<F> & proto, std::vector<F> & value)
    {
        if (proto.column.size() != value.size())
        {
            throw std::invalid_argument(std::format("{}: Mismatch between prototype's column and value's size", __FUNCTION__));
        }

        csr.rows = proto.rows;
        csr.cols = proto.cols;
        csr.rowStart = proto.rowStart;
        csr.column = proto.column;
        csr.values = std::move(value);
    }

    template <typename F>
    ChorinCsrMatrices<F> buildChorinCsrMatrices(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                                                const int integrationDegree, const int nThreads)
    {
        if (velocityMesh.numElements != pressureMesh.numElements)
        {
            throw std::invalid_argument(std::format("{}: Velocity and pressure meshes have different number of elements", __FUNCTION__));
        }
        if (nThreads < 1 || nThreads > 32)
        {
            throw std::invalid_argument(std::format("{}: Bad number of threads [{}]", __FUNCTION__, nThreads));
        }

        // Build prototype matrices first - structure only
        // u::Stopwatch sw;
        const auto proto = buildPrototypes4T<F>(velocityMesh, pressureMesh);
        // const auto tProto = sw.millis(true);

        std::vector<ValueBundle<F>> bundles(nThreads);
        for (int i = 0; i < nThreads; i++)
        {
            bundles[i].velocityMass.resize(proto.velocity.column.size());
            bundles[i].velocityStiffness.resize(proto.velocity.column.size());
            bundles[i].pressureStiffness.resize(proto.pressure.column.size());
            bundles[i].velocityPressureDiv.resize(proto.velocityPressureDiv.column.size());
            bundles[i].pressureVelocityDiv.resize(proto.pressureVelocityDiv.column.size());
        }

        // const auto tAlloc = sw.millis(true);

        const int nExtraThreads = nThreads - 1;
        std::vector<std::future<void>> extraThreads(nExtraThreads);
        std::atomic<int> counter(0);

        for (int i = 0; i < nExtraThreads; i++)
        {
            extraThreads[i] = std::async(std::launch::async, buildPart<F>,
                                         std::ref(proto), std::ref(bundles[i + 1]),
                                         std::ref(velocityMesh), std::ref(pressureMesh),
                                         std::ref(integrationDegree), std::ref(counter));
        }
        buildPart<F>(proto, bundles[0], velocityMesh, pressureMesh, integrationDegree, counter);

        // const auto tParts = sw.millis(true);

        for (int i = 0; i < nExtraThreads; i++)
        {
            extraThreads[i].get();
            bundles[0] += bundles[i + 1];
        }

        // const auto tSum = sw.millis(true);

        ChorinCsrMatrices<F> result;
        assembleCsrPrototypeValue(result.velocityMass, proto.velocity, bundles[0].velocityMass);
        assembleCsrPrototypeValue(result.velocityStiffness, proto.velocity, bundles[0].velocityStiffness);
        assembleCsrPrototypeValue(result.pressureStiffness, proto.pressure, bundles[0].pressureStiffness);
        assembleCsrPrototypeValue(result.velocityPressureDiv, proto.velocityPressureDiv, bundles[0].velocityPressureDiv);
        assembleCsrPrototypeValue(result.pressureVelocityDiv, proto.pressureVelocityDiv, bundles[0].pressureVelocityDiv);

        // const auto tMove = sw.millis();

        // std::cout << std::format("Times: proto = {}, alloc = {}, parts = {}, sum = {}, move = {}\n",
        //                          tProto, tAlloc, tParts, tSum, tMove);

        return result;
    }

    template ChorinCsrMatrices<float> buildChorinCsrMatrices(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                                                             const int integrationDegree, const int nThreads);
} // namespace fem