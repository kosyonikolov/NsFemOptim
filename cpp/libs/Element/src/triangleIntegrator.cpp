#include <element/triangleIntegrator.h>

#include <array>
#include <format>
#include <stdexcept>
// #include <iostream>

namespace el
{
    std::vector<IntegrationPoint> getIntegrationPoints(const int degree)
    {
        // From https://www.math.unipd.it/~alvise/SETS_CUBATURE_TRIANGLE/cowper/set_cowper_standard.m
        if (degree == 2)
        {
            return {IntegrationPoint{5.00000000000000000000000000000000e-01, 5.00000000000000000000000000000000e-01, 1.66666666666666657414808128123695e-01},
                    IntegrationPoint{5.00000000000000000000000000000000e-01, 0.00000000000000000000000000000000e+00, 1.66666666666666657414808128123695e-01},
                    IntegrationPoint{0.00000000000000000000000000000000e+00, 5.00000000000000000000000000000000e-01, 1.66666666666666657414808128123695e-01}};
        }
        if (degree == 3)
        {

            return {IntegrationPoint{3.33333333333333314829616256247391e-01, 3.33333333333333314829616256247391e-01, -2.81250000000000000000000000000000e-01},
                    IntegrationPoint{2.00000000000000011102230246251565e-01, 2.00000000000000011102230246251565e-01, 2.60416666666666685170383743752609e-01},
                    IntegrationPoint{2.00000000000000011102230246251565e-01, 5.99999999999999977795539507496869e-01, 2.60416666666666685170383743752609e-01},
                    IntegrationPoint{5.99999999999999977795539507496869e-01, 2.00000000000000011102230246251565e-01, 2.60416666666666685170383743752609e-01}};
        }
        if (degree == 4)
        {
            return {IntegrationPoint{9.15762135097710067155318824916321e-02, 9.15762135097710067155318824916321e-02, 5.49758718276610602870846378209535e-02},
                    IntegrationPoint{9.15762135097710067155318824916321e-02, 8.16847572980457958813360619387822e-01, 5.49758718276610602870846378209535e-02},
                    IntegrationPoint{8.16847572980457958813360619387822e-01, 9.15762135097710067155318824916321e-02, 5.49758718276610602870846378209535e-02},
                    IntegrationPoint{4.45948490915965001235576892213430e-01, 4.45948490915965001235576892213430e-01, 1.11690794839005624883299105931656e-01},
                    IntegrationPoint{4.45948490915965001235576892213430e-01, 1.08103018168069997528846215573139e-01, 1.11690794839005624883299105931656e-01},
                    IntegrationPoint{1.08103018168069997528846215573139e-01, 4.45948490915965001235576892213430e-01, 1.11690794839005624883299105931656e-01}};
        }
        if (degree == 5)
        {
            return {IntegrationPoint{3.33333333333333314829616256247391e-01, 3.33333333333333314829616256247391e-01, 1.12500000000000002775557561562891e-01},
                    IntegrationPoint{1.01286507323456329010546994595643e-01, 1.01286507323456329010546994595643e-01, 6.29695902724135836425745083033689e-02},
                    IntegrationPoint{1.01286507323456329010546994595643e-01, 7.97426985353087314223330395179801e-01, 6.29695902724135836425745083033689e-02},
                    IntegrationPoint{7.97426985353087314223330395179801e-01, 1.01286507323456329010546994595643e-01, 6.29695902724135836425745083033689e-02},
                    IntegrationPoint{4.70142064105115053962435922585428e-01, 4.70142064105115053962435922585428e-01, 6.61970763942530820989063045090006e-02},
                    IntegrationPoint{4.70142064105115053962435922585428e-01, 5.97158717897698920751281548291445e-02, 6.61970763942530820989063045090006e-02},
                    IntegrationPoint{5.97158717897698920751281548291445e-02, 4.70142064105115053962435922585428e-01, 6.61970763942530820989063045090006e-02}};
        }

        throw std::invalid_argument(std::format("{}: No formula for degree [{}]", __FUNCTION__, degree));
    }

    std::vector<IntegrationPoint> getLineIntegrationPoints(const int degree)
    {
        // Gauss quadrature, scaled and shifted to operate on [0, 1] instead of [-1, 1]
        if (degree == 2)
        {
            return {IntegrationPoint{0.5, 0, 0.211324865405187},
                    IntegrationPoint{0.5, 0, 0.788675134594813}};
        }
        else if (degree == 3)
        {
            return {IntegrationPoint{0.277777777777778, 0, 0.112701665379258},
                    IntegrationPoint{0.444444444444444, 0, 0.5},
                    IntegrationPoint{0.277777777777778, 0, 0.887298334620742}};
        }
        else if (degree == 4)
        {
            return {IntegrationPoint{0.173927422568727, 0, 0.0694318442029737},
                    IntegrationPoint{0.326072577431273, 0, 0.330009478207572},
                    IntegrationPoint{0.326072577431273, 0, 0.669990521792428},
                    IntegrationPoint{0.173927422568727, 0, 0.930568155797026}};
        }
        else if (degree == 5)
        {
            return {IntegrationPoint{0.118463442528095, 0, 0.046910077030668},
                    IntegrationPoint{0.239314335249683, 0, 0.230765344947158},
                    IntegrationPoint{0.284444444444444, 0, 0.5},
                    IntegrationPoint{0.239314335249683, 0, 0.769234655052842},
                    IntegrationPoint{0.118463442528095, 0, 0.953089922969332}};
        }

        throw std::invalid_argument(std::format("{}: No formula for degree [{}]", __FUNCTION__, degree));
    }

    Mat22 calcJacobian(const AffineTransform & t)
    {
        Mat22 j;

        j[0][0] = t.m[0][0];
        j[0][1] = t.m[1][0];
        j[1][0] = t.m[0][1];
        j[1][1] = t.m[1][1];

        return j;
    }

    cv::Mat calcB(const AffineTransform & t)
    {
        cv::Mat b(2, 2, CV_32FC1);

        b.at<float>(0, 0) = t.m[1][1];
        b.at<float>(0, 1) = -t.m[1][0];
        b.at<float>(1, 0) = -t.m[0][1];
        b.at<float>(1, 1) = t.m[0][0];

        return b;
    }

    float det(const Mat22 & m)
    {
        return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    }

    TriangleIntegrator::TriangleIntegrator(const Element & element, const int degree)
        : element(element)
    {
        nDof = dof(element.type);
        shapeFn = getShapeFunction(element.type);
        shapeGradFn = getShapeGradFunction(element.type);
        valueFn = getValueFunction(element.type);

        intPts = getIntegrationPoints(degree);
        lineIntPts = getLineIntegrationPoints(degree);

        phi.resize(nDof);
        grad.create(2, nDof, CV_32FC1);
    }

    void TriangleIntegrator::integrateLocalMassMatrix(const AffineTransform & t, cv::Mat & dst) const
    {
        dst.create(nDof, nDof, CV_32FC1);
        dst.setTo(0);

        for (const auto [x, y, w] : intPts)
        {
            shapeFn(x, y, phi.data());
            for (int i = 0; i < nDof; i++)
            {
                float * row = dst.ptr<float>(i);
                // Accumulate only lower triangle
                for (int j = 0; j <= i; j++)
                {
                    row[j] += phi[i] * phi[j] * w;
                }
            }
        }

        // Copy values to upper triangle
        for (int i = 0; i < nDof; i++)
        {
            for (int j = i + 1; j < nDof; j++)
            {
                dst.at<float>(i, j) = dst.at<float>(j, i);
            }
        }

        const auto j = calcJacobian(t);
        const float absDetJ = std::abs(det(j));
        dst *= absDetJ;
    }

    void TriangleIntegrator::integrateLocalStiffnessMatrix(const AffineTransform & t, cv::Mat & dst) const
    {
        dst.create(nDof, nDof, CV_32FC1);
        dst.setTo(0);

        const auto j = calcJacobian(t);
        const float invAbsDetJ = 1.0f / std::abs(det(j));

        const auto b = calcB(t);
        const auto btb = b.t() * b;

        // std::cout << "absDetJ: " << absDetJ << "\n";
        // std::cout << "B:\n" << b << "\n";
        // std::cout << "BTB:\n" << btb << "\n";

        float * gradX = grad.ptr<float>(0);
        float * gradY = grad.ptr<float>(1);
        for (const auto [x, y, w] : intPts)
        {
            shapeGradFn(x, y, gradX, gradY);
            const float totalW = w * invAbsDetJ;
            const cv::Mat contrib = grad.t() * btb * grad * totalW;
            // std::cout << contrib << "\n";
            dst += contrib;
        }
    }
} // namespace el