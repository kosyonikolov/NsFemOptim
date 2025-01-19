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
        ValueFn valueFn;

        std::vector<IntegrationPoint> intPts;

        // Alloc-once buffers
        mutable std::vector<float> phi;

    public:
        TriangleIntegrator(const Element & element, const int degree);

        void integrateLocalMassMatrix(const AffineTransform & t, cv::Mat & dst) const;

        template<typename F>
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
    };
}; // namespace el

#endif /* LIBS_ELEMENT_INCLUDE_ELEMENT_TRIANGLEINTEGRATOR */
