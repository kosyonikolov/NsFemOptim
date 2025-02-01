#ifndef LIBS_ELEMENT_INCLUDE_ELEMENT_TRIANGLEINTEGRATOR
#define LIBS_ELEMENT_INCLUDE_ELEMENT_TRIANGLEINTEGRATOR

#include <vector>

#include <opencv2/opencv.hpp>

#include <element/affineTransform.h>
#include <element/calc.h>
#include <element/element.h>

namespace el
{
    using Mat22 = std::array<std::array<float, 2>, 2>;

    Mat22 calcJacobian(const AffineTransform & t);

    float det(const Mat22 & m);

    cv::Mat calcB(const AffineTransform & t);

    struct IntegrationPoint
    {
        float x;
        float y;
        float w;
    };

    class TriangleIntegrator
    {
        Element element;

        int nDof;
        ShapeFn shapeFn;
        ShapeGradFn shapeGradFn;
        ValueFn valueFn;

        // 2D points on reference triangle
        std::vector<IntegrationPoint> intPts;

        // 1D points on [0, 1] - y component is zero
        std::vector<IntegrationPoint> lineIntPts;

        // Alloc-once buffers
        mutable std::vector<float> phi;
        mutable cv::Mat grad;                   // shape = (2, DOF), first row = gradX, second row = gradY
        mutable std::vector<float> gradFlowDot; // DOF

    public:
        TriangleIntegrator(const Element & element, const int degree);

        void integrateLocalMassMatrix(const AffineTransform & t, cv::Mat & dst) const;

        void integrateLocalStiffnessMatrix(const AffineTransform & t, cv::Mat & dst) const;

        template <typename F>
        void integrateLocalLoadVector(const AffineTransform & t, F func, std::vector<float> & dst)
        {
            dst.resize(nDof);
            std::fill(dst.begin(), dst.end(), 0);

            const auto j = calcJacobian(t);
            const float absDetJ = std::abs(det(j));

            for (const auto [x, y, w] : intPts)
            {
                shapeFn(x, y, phi.data());
                const auto globalPt = t(Point{x, y});
                const float v = func(globalPt.x, globalPt.y);
                const float totalW = w * absDetJ * v;
                for (int i = 0; i < nDof; i++)
                {
                    dst[i] += phi[i] * totalW;
                }
            }
        }

        template <typename F>
        void integrateLocalBorderLoadVector(const AffineTransform & tFwd, F flowFunc, const int side, std::vector<float> & dst)
        {
            assert(side >= 0 && side < 6);
            const int sideId = side / 2;

            constexpr std::array<Point, 3> refPts = {Point{0, 0}, Point{1, 0}, Point{0, 1}};
            const Point refStart = refPts[sideId];
            const Point refEnd = refPts[(sideId + 1) % 3];

            // Calculate length of side in global triangle
            const auto globalStart = tFwd(refStart);
            const auto globalEnd = tFwd(refEnd);
            // const float globalSideLen = distance(globalStart, globalEnd);
            // Calculate global normal
            // Do not normalize it - this avoids an additional multiplication by its length
            Point globalNormal = {globalEnd.y - globalStart.y, -(globalEnd.x - globalStart.x)};

            dst.resize(nDof);
            std::fill(dst.begin(), dst.end(), 0);

            for (const auto [x, _, w] : lineIntPts)
            {
                const float cX = 1 - x;
                const Point refPt{cX * refStart.x + x * refEnd.x, cX * refStart.y + x * refEnd.y};
                const Point globalPt = tFwd(refPt);
                const Point globalFlow = flowFunc(globalPt);
                const float normalFlow = globalFlow.x * globalNormal.x + globalFlow.y * globalNormal.y;
                const float totalW = w * normalFlow;

                shapeFn(refPt.x, refPt.y, phi.data());
                for (int i = 0; i < nDof; i++)
                {
                    dst[i] += phi[i] * totalW;
                }
            }
        }

        // Local convection matrix for generic flow function
        template <typename F>
        void integrateLocalConvectionMatrix(const AffineTransform & tFwd, F flowFunc, cv::Mat & dst)
        {
            dst.create(nDof, nDof, CV_32FC1);
            dst.setTo(0);

            const auto j = calcJacobian(tFwd);
            const float jSign = det(j) < 0 ? -1 : 1;
            const auto b = calcB(tFwd);

            float * gradX = grad.ptr<float>(0);
            float * gradY = grad.ptr<float>(1);
            for (const auto [x, y, w] : intPts)
            {
                shapeFn(x, y, phi.data());
                shapeGradFn(x, y, gradX, gradY);
                const auto globalPt = tFwd(Point{x, y});
                const Point globalFlow = flowFunc(globalPt);
                const Point localFlow{b.at<float>(0, 0) * globalFlow.x + b.at<float>(1, 0) * globalFlow.y,
                                      b.at<float>(0, 1) * globalFlow.x + b.at<float>(1, 1) * globalFlow.y};

                for (int i = 0; i < nDof; i++)
                {
                    gradFlowDot[i] = localFlow.x * grad.at<float>(0, i) + localFlow.y * grad.at<float>(1, i);
                }
                const float totalW = w * jSign;
                for (int r = 0; r < nDof; r++)
                {
                    float * dstRow = dst.ptr<float>(r);
                    for (int c = 0; c < nDof; c++)
                    {
                        dstRow[c] += gradFlowDot[c] * phi[r] * totalW;
                    }
                }
            }
        }

        // Local convection matrix for flow that is defined on the same element
        // flowX and flowY must be DOF-sized vectors with the flow velocities
        void integrateLocalSelfConvectionMatrix(const AffineTransform & tFwd, const float * flowX, const float * flowY, cv::Mat & dst);
    };
}; // namespace el

#endif /* LIBS_ELEMENT_INCLUDE_ELEMENT_TRIANGLEINTEGRATOR */
