#include <fem/fastConvection.h>

// #include <utils/stopwatch.h>

namespace fem
{
    using Triplet = Eigen::Triplet<float>;

    FastConvection::FastConvection(const mesh::ConcreteMesh & velocityMesh, el::TriangleIntegrator & integrator)
    {
        const int nNodes = velocityMesh.nodes.size();
        const int nElems = velocityMesh.numElements;
        const int elSize = velocityMesh.getElementSize();

        // u::Stopwatch sw;
        convection = Eigen::SparseMatrix<float, Eigen::RowMajor>(nNodes, nNodes);

        // Assemble convection with fake data to create the sparse pattern
        std::vector<int> ids(elSize);
        std::vector<Triplet> fakeTriplets;
        for (int i = 0; i < nElems; i++)
        {
            velocityMesh.getElement(i, ids.data(), 0);
            for (int r = 0; r < elSize; r++)
            {
                const int globalRow = ids[r];
                for (int c = 0; c < elSize; c++)
                {
                    const int globalCol = ids[c];
                    fakeTriplets.emplace_back(globalRow, globalCol, 1);
                }
            }
        }

        convection.setFromTriplets(fakeTriplets.begin(), fakeTriplets.end());
        assert(convection.isCompressed());

        // const auto tFakeConvection = sw.millis(true);

        const int nnz = convection.nonZeros();
        auto pVal = convection.valuePtr();

        // Construct the integration matrix
        // It has size E x 2N, where E = nnz (nonzero entries in convection)
        integration = Eigen::SparseMatrix<float, Eigen::RowMajor>(nnz, 2 * nNodes);
        std::vector<float> localVx(elSize, 0);
        std::vector<float> localVy(elSize, 0);
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> rowIndices(elSize, elSize);
        cv::Mat localConvection;
        std::vector<Triplet> integrationTriplets;
        for (int i = 0; i < nElems; i++)
        {
            velocityMesh.getElement(i, ids.data(), 0);
            // The ids determine the E-idx (row idx)
            // Calculate them once
            for (int r = 0; r < elSize; r++)
            {
                const int globalRow = ids[r];
                for (int c = 0; c < elSize; c++)
                {
                    const int globalCol = ids[c];
                    auto & ref = convection.coeffRef(globalRow, globalCol);
                    const int idx = &ref - pVal;
                    assert(idx >= 0 && idx < nnz);
                    rowIndices(r, c) = idx;
                }
            }

            // Set each velocity component to 1 and calculate the contribution
            for (int iVxy = 0; iVxy < 2 * elSize; iVxy++)
            {
                const bool isVy = iVxy >= elSize;
                std::vector<float> & localVelocity = isVy ? localVy : localVx;
                const int localIdx = isVy ? iVxy - elSize : iVxy;
                localVelocity[localIdx] = 1;
                integrator.integrateLocalSelfConvectionMatrix(velocityMesh.elementTransforms[i], localVx.data(), localVy.data(), localConvection);
                localVelocity[localIdx] = 0;

                // Accumulate to integration matrix
                const int globalNodeIdx = ids[localIdx];
                const int colIdx = isVy ? globalNodeIdx + nNodes : globalNodeIdx;
                for (int r = 0; r < elSize; r++)
                {
                    for (int c = 0; c < elSize; c++)
                    {
                        const int rowIdx = rowIndices(r, c);
                        const float val = localConvection.at<float>(r, c);
                        integrationTriplets.emplace_back(rowIdx, colIdx, val);
                    }
                }
            }
        }
        integration.setFromTriplets(integrationTriplets.begin(), integrationTriplets.end());

        // const auto tIntegration = sw.millis();
        // std::cout << "Fast integration times: fake = " << tFakeConvection << " ms, integration = " << tIntegration << " ms\n";

        velocity = Eigen::Vector<float, Eigen::Dynamic>(2 * nNodes);
        values = Eigen::Vector<float, Eigen::Dynamic>(nnz);
    }

    void FastConvection::update(const std::vector<float> & velocityXy)
    {
        const int n = velocity.rows();
        assert(n == velocityXy.size());
        for (int i = 0; i < n; i++)
        {
            velocity[i] = velocityXy[i];
        }

        values = integration * velocity;
        const int nnz = convection.nonZeros();
        assert(values.rows() == nnz);
        std::copy_n(values.data(), nnz, convection.valuePtr());
    }
} // namespace fem