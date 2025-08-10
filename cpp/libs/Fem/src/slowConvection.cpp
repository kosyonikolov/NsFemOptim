#include <fem/slowConvection.h>

#include <utils/stopwatch.h>

namespace fem
{
    using Triplet = Eigen::Triplet<float>;

    SlowConvection::SlowConvection(const mesh::ConcreteMesh & velocityMesh, el::TriangleIntegrator & integrator)
        : velocityMesh(velocityMesh), integrator(integrator)
    {
        // Initialize the convection matrix
        const int nNodes = velocityMesh.nodes.size();
        const int nElems = velocityMesh.numElements;
        const int elSize = velocityMesh.getElementSize();

        u::Stopwatch sw;

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

        // const auto tConstruct = sw.millis(true);

        // Extract offsets in the value vector for all elements
        elementIndices.resize(nElems);
        float * values = convection.valuePtr();
        [[maybe_unused]] const int nnz = convection.nonZeros();
        for (int i = 0; i < nElems; i++)
        {
            auto & m = elementIndices[i];
            m.create(elSize, elSize, CV_32S);
            velocityMesh.getElement(i, ids.data(), 0);

            for (int r = 0; r < elSize; r++)
            {
                const int globalRow = ids[r];
                for (int c = 0; c < elSize; c++)
                {
                    const int globalCol = ids[c];
                    auto & ref = convection.coeffRef(globalRow, globalCol);
                    const int idx = &ref - values;
                    assert(idx >= 0 && idx < nnz);
                    m.at<int>(r, c) = idx;
                }
            }
        }

        // const auto tIndices = sw.millis();
        // std::cout << "Construct = " << tConstruct << " us, indices = " << tIndices << " us\n";
    }

    void SlowConvection::update(const std::vector<float> & velocityXy)
    {
        const int nNodes = velocityMesh.nodes.size();
        const int nElems = velocityMesh.numElements;
        const int elSize = velocityMesh.getElementSize();

        assert(velocityXy.size() == 2 * nNodes);
        assert(convection.rows() == nNodes);
        assert(convection.cols() == nNodes);
        assert(convection.isCompressed());

        float * values = convection.valuePtr();
        const int nnz = convection.nonZeros();
        std::fill_n(values, nnz, 0.0f);

        std::vector<float> localVx(elSize, 0);
        std::vector<float> localVy(elSize, 0);
        cv::Mat localConvection;
        std::vector<int> ids(elSize);

        for (int i = 0; i < nElems; i++)
        {
            velocityMesh.getElement(i, ids.data(), 0);
         
            // Extract the local velocity
            for (int j = 0; j < elSize; j++)
            {
                const int globalId = ids[j];
                localVx[j] = velocityXy[globalId];
                localVy[j] = velocityXy[globalId + nNodes];
            }

            // u::Stopwatch sw;

            // Compute the local convection
            integrator.integrateLocalSelfConvectionMatrix(velocityMesh.elementTransforms[i], localVx.data(), localVy.data(), localConvection);

            // const auto tLocal = 1000 * sw.millis(true);

            // Accumulate the local matrix
            const auto & indexM = elementIndices[i];
            for (int r = 0; r < elSize; r++)
            {
                const float * localRow = localConvection.ptr<float>(r);
                const int * indexRow = indexM.ptr<int>(r);
                for (int c = 0; c < elSize; c++)
                {
                    const int idx = indexRow[c];
                    const float val = localRow[c];
                    values[idx] += val;
                }
            }

            // const auto tAcc = 1000 * sw.millis();
            // std::cout << "Local = " << tLocal << " us, acc = " << tAcc << "\n";
        }
    }
} // namespace fem