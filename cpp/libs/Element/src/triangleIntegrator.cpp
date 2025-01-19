#include <element/triangleIntegrator.h>

#include <array>
#include <stdexcept>
#include <format>

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

    Mat22 calcJacobian(const AffineTransform & t)
    {
        Mat22 j;

        j[0][0] = t.m[0][0];
        j[0][1] = t.m[1][0];
        j[1][0] = t.m[0][1];
        j[1][1] = t.m[1][1];

        return j;
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
        valueFn = getValueFunction(element.type);

        intPts = getIntegrationPoints(degree);

        phi.resize(nDof);
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
} // namespace el