#include <fem/chorinMatrices.h>

#include <stdexcept>
#include <format>

#include <element/triangleIntegrator.h>

namespace fem
{
    template <typename F>
    ChorinMatrices<F> buildChorinMatrices(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                                          const int integrationDegree)
    {
        if (velocityMesh.numElements != pressureMesh.numElements)
        {
            throw std::invalid_argument(std::format("{}: Velocity and pressure meshes have different number of elements", __FUNCTION__));
        }

        const int numVelocityNodes = velocityMesh.nodes.size(); // ! Counts only one channel, not X + Y - multiply by 2 to get all velocity nodes
        const int numPressureNodes = pressureMesh.nodes.size();

        ChorinMatrices<F> result;
        result.velocityMass.resize(numVelocityNodes, numVelocityNodes);
        result.velocityStiffness.resize(numVelocityNodes, numVelocityNodes);
        result.pressureStiffness.resize(numPressureNodes, numPressureNodes);
        result.velocityPressureDiv.resize(numPressureNodes, 2 * numVelocityNodes);
        result.pressureVelocityDiv.resize(2 * numVelocityNodes, numPressureNodes);

        el::TriangleIntegrator velocityIntegrator(velocityMesh.baseElement, integrationDegree, pressureMesh.baseElement);
        el::TriangleIntegrator pressureIntegrator(pressureMesh.baseElement, integrationDegree);

        const int elSizeV = velocityMesh.getElementSize();
        const int elSizeP = pressureMesh.getElementSize();

        std::vector<int> idsV(elSizeV);
        std::vector<el::Point> ptsV(elSizeV);
        cv::Mat localMassMatrixV, localStiffnessMatrixV;

        std::vector<int> idsP(elSizeP);
        std::vector<el::Point> ptsP(elSizeP);
        cv::Mat localMassMatrixP, localStiffnessMatrixP;
        cv::Mat localVpdX, localVpdY, localPvdX, localPvdY;

        const int nElem = velocityMesh.numElements;
        for (int i = 0; i < nElem; i++)
        {
            const auto t = velocityMesh.elementTransforms[i];

            // ========================= Velocity-only matrices =========================
            velocityMesh.getElement(i, idsV.data(), ptsV.data());

            velocityIntegrator.integrateLocalMassMatrix(t, localMassMatrixV);
            assert(elSizeV == localMassMatrixV.cols);
            assert(elSizeV == localMassMatrixV.rows);

            velocityIntegrator.integrateLocalStiffnessMatrix(t, localStiffnessMatrixV);
            assert(elSizeV == localStiffnessMatrixV.cols);
            assert(elSizeV == localStiffnessMatrixV.rows);

            // Accumulate
            for (int r = 0; r < elSizeV; r++)
            {
                const int i = idsV[r];
                for (int c = 0; c < elSizeV; c++)
                {
                    const int j = idsV[c];
                    result.velocityMass.add(i, j, localMassMatrixV.at<float>(r, c));
                    result.velocityStiffness.add(i, j, localStiffnessMatrixV.at<float>(r, c));
                    // const Triplet tripletM(i, j, localMassMatrixV.at<float>(r, c));
                    // const Triplet tripletS(i, j, localStiffnessMatrixV.at<float>(r, c));
                    // velocityMassT.push_back(tripletM);
                    // velocityStiffnessT.push_back(tripletS);
                }
            }

            // ========================= Pressure-only matrices =========================
            pressureMesh.getElement(i, idsP.data(), ptsP.data());

            pressureIntegrator.integrateLocalStiffnessMatrix(t, localStiffnessMatrixP);
            assert(elSizeP == localStiffnessMatrixP.cols);
            assert(elSizeP == localStiffnessMatrixP.rows);

            // Accumulate
            for (int r = 0; r < elSizeP; r++)
            {
                const int i = idsP[r];
                for (int c = 0; c < elSizeP; c++)
                {
                    const int j = idsP[c];
                    result.pressureStiffness.add(i, j, localStiffnessMatrixP.at<float>(r, c));
                    // const Triplet tripletS(i, j, localStiffnessMatrixP.at<float>(r, c));
                    // pressureStiffnessT.push_back(tripletS);
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

            for (int iV = 0; iV < elSizeV; iV++)
            {
                const int gV = idsV[iV];
                for (int iP = 0; iP < elSizeP; iP++)
                {
                    const int gP = idsP[iP];
                    // clang-format off
                    result.velocityPressureDiv.add(gP, gV,                    localVpdX.at<float>(iP, iV));
                    result.velocityPressureDiv.add(gP, gV + numVelocityNodes, localVpdY.at<float>(iP, iV));
                    result.pressureVelocityDiv.add(gV, gP,                    localPvdX.at<float>(iV, iP));
                    result.pressureVelocityDiv.add(gV + numVelocityNodes, gP, localPvdY.at<float>(iV, iP));
                    // velocityPressureDivT.emplace_back(gP, gV,                    localVpdX.at<float>(iP, iV));
                    // velocityPressureDivT.emplace_back(gP, gV + numVelocityNodes, localVpdY.at<float>(iP, iV));
                    // pressureVelocityDivT.emplace_back(gV, gP,                    localPvdX.at<float>(iV, iP));
                    // pressureVelocityDivT.emplace_back(gV + numVelocityNodes, gP, localPvdY.at<float>(iV, iP));
                    // clang-format on
                }
            }
        }

        return result;
    }

    template ChorinMatrices<float> buildChorinMatrices(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                                                       const int integrationDegree);
} // namespace fem