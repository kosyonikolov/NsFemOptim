#include <fem/verySlowConvection.h>

namespace fem
{
    using Triplet = Eigen::Triplet<float>;

    VerySlowConvection::VerySlowConvection(const mesh::ConcreteMesh & velocityMesh, el::TriangleIntegrator & integrator)
        : velocityMesh(velocityMesh), integrator(integrator)
    {
    }

    Eigen::SparseMatrix<float, Eigen::RowMajor> VerySlowConvection::calculate(const std::vector<float> & velocityXy)
    {
        const int nNodes = velocityMesh.nodes.size();
        const int nElems = velocityMesh.numElements;
        const int elSize = velocityMesh.getElementSize();

        Eigen::SparseMatrix<float, Eigen::RowMajor> convection = Eigen::SparseMatrix<float, Eigen::RowMajor>(nNodes, nNodes);

        std::vector<int> ids(elSize);
        std::vector<Triplet> triplets;
        std::vector<float> localVx(elSize, 0);
        std::vector<float> localVy(elSize, 0);
        cv::Mat localConvection;

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

            // Compute the local convection
            integrator.integrateLocalSelfConvectionMatrix(velocityMesh.elementTransforms[i], localVx.data(), localVy.data(), localConvection);

            // Accumulate the local matrix
            for (int r = 0; r < elSize; r++)
            {
                const int globalRow = ids[r];
                const float * localRow = localConvection.ptr<float>(r);
                for (int c = 0; c < elSize; c++)
                {
                    const int globalCol = ids[c];
                    const float val = localRow[c];
                    triplets.emplace_back(globalRow, globalCol, val);
                }
            }
        }

        convection.setFromTriplets(triplets.begin(), triplets.end());
        assert(convection.isCompressed());

        return convection;
    }
} // namespace fem