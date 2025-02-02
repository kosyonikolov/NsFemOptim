#include <mesh/drawMesh.h>

#include <limits>
#include <cassert>
#include <span>

#include <mesh/interpolator.h>
#include <mesh/colorScale.h>
#include <mesh/triangleLookup.h>

namespace mesh
{
    cv::Mat drawMesh(const ConcreteMesh & mesh, const float scale)
    {
        // Find min/max XY values
        constexpr float inf = std::numeric_limits<float>::infinity();
        float minX = inf;
        float maxX = -inf;
        float minY = inf;
        float maxY = -inf;
        for (const auto & pt : mesh.nodes)
        {
            minX = std::min(minX, pt.x);
            maxX = std::max(maxX, pt.x);
            minY = std::min(minY, pt.y);
            maxY = std::max(maxY, pt.y);
        }

        const float spanX = maxX - minX;
        const float spanY = maxY - minY;
        const int borderPx = 100;
        
        auto cvPoint = [&](const el::Point & p)
        {
            const int x = borderPx + scale * (p.x - minX);
            const int y = borderPx + scale * (p.y - minY);
            return cv::Point2i(x, y);
        };

        const int width = 2 * borderPx + spanX * scale;
        const int height = 2 * borderPx + spanY * scale;
        cv::Mat result = cv::Mat::zeros(height, width, CV_8UC3);

        // Draw triangles
        const cv::Scalar triColor(0, 0, 255);
        std::array<el::Point, 3> refPts = {el::Point{0,0}, el::Point{0,1}, el::Point{1,0}};
        std::array<cv::Point2i, 3> cvTriPts;
        const int nElem = mesh.numElements;
        for (int i = 0; i < nElem; i++)
        {
            const auto & t = mesh.elementTransforms[i];
            for (int k = 0; k < 3; k++)
            {
                auto pt = t(refPts[k]);
                cvTriPts[k] = cvPoint(pt);
            }

            cv::line(result, cvTriPts[0], cvTriPts[1], triColor);
            cv::line(result, cvTriPts[1], cvTriPts[2], triColor);
            cv::line(result, cvTriPts[2], cvTriPts[0], triColor);
        }

        // Draw borders
        const std::vector<cv::Scalar> borderColors
        {
            cv::Scalar(0, 255, 0),
            cv::Scalar(255, 0, 0),
            cv::Scalar(255, 0, 255),
            cv::Scalar(0, 255, 255),
            cv::Scalar(255, 255, 0),
            cv::Scalar(0, 128, 0),
            cv::Scalar(128, 0, 0),
            cv::Scalar(128, 0, 128)
        };

        const int nBorderElements = mesh.numBorderElements;
        const int borderElSize = mesh.getBorderElementSize();
        std::vector<el::Point> borderPts(borderElSize);
        for (int i = 0; i < nBorderElements; i++)
        {
            int tri, side, group;
            mesh.getBorderElement(i, tri, side, group, 0, borderPts.data());
            const auto color = borderColors[group % borderColors.size()];
            for (int j = 0; j < borderElSize - 1; j++)
            {
                auto a = cvPoint(borderPts[j]);
                auto b = cvPoint(borderPts[j + 1]);
                auto delta = a - b;
                const float len = std::sqrt(delta.x * delta.x + delta.y * delta.y);
                cv::arrowedLine(result, a, b, color, 2, 8, 0, 10 / len);
            }
        }

        // Draw nodes
        const cv::Scalar nodeColor(128, 128, 128);
        const cv::Scalar textColor(255, 255, 255);
        const int nPts = mesh.nodes.size();
        for (int i = 0; i < nPts; i++)
        {
            const auto pt = cvPoint(mesh.nodes[i]);
            cv::circle(result, pt, 2, nodeColor, cv::FILLED);
            cv::putText(result, std::to_string(i), pt, cv::FONT_HERSHEY_SIMPLEX, 0.5, textColor);
        }

        return result;
    }

    cv::Mat drawValues(const Interpolator & interpolator, const AbstractColorScale & colorScale, const float scale)
    {
        const auto iSize = interpolator.getRange();
        const int width = iSize.width * scale + 1;
        const int height = iSize.height * scale + 1;
        cv::Mat result = cv::Mat::zeros(height, width, CV_8UC3);

        const float invS = 1.0f / scale;
        for (int iy = 0; iy < height; iy++)
        {
            const float y = (height - 1 - iy) * invS + iSize.minY;
            uint8_t * line = result.ptr<uint8_t>(iy);
            for (int ix = 0; ix < width; ix++)
            {
                const float x = ix * invS + iSize.minX;
                const auto val = interpolator.interpolate(x, y);
                if (!val)
                {
                    continue;
                } 
                const auto color = colorScale(val.value());
               
                uint8_t * pix = line + 3 * ix;
                pix[0] = color[0];
                pix[1] = color[1];
                pix[2] = color[2];
            }
        }

        return result;
    }

    cv::Mat drawCfd(const TriangleLookup & triangleLookup, const AbstractColorScale & pressureScale,
                    const float imgScale0, const float velocityScale, const float velocityStep,
                    const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                    const std::vector<float> & velocityXy, const std::vector<float> & pressure)
    {
        const int width = 2 *  ((static_cast<int>(triangleLookup.width * imgScale0 + 0.5) + 1) / 2);
        const int height = 2 * ((static_cast<int>(triangleLookup.height * imgScale0 + 0.5) + 1) / 2);
        const float imgScale = width / triangleLookup.width;
        const float invS = 1.0f / imgScale;

        cv::Mat result = cv::Mat::zeros(height, width, CV_8UC3);

        const int numVelocityNodes = velocityMesh.nodes.size();
        [[maybe_unused]] const int numPressureNodes = pressureMesh.nodes.size();

        assert(velocityXy.size() == 2 * numVelocityNodes);
        assert(pressure.size() == numPressureNodes);

        const std::span<const float> vx(velocityXy.data(), numVelocityNodes);
        const std::span<const float> vy(velocityXy.data() + numVelocityNodes, numVelocityNodes);

        const int elSizeV = velocityMesh.getElementSize();
        const int elSizeP = pressureMesh.getElementSize();

        std::vector<int> velocityIds(elSizeV);
        std::vector<int> pressureIds(elSizeP);

        std::vector<float> localVx(elSizeV), localVy(elSizeV);
        std::vector<float> localPressure(elSizeP);

        // Draw pressure
        auto pressureValue = el::getValueFunction(pressureMesh.baseElement.type);
        assert(pressureValue);
        int lastTriangleId = 0;
        for (int iy = 0; iy < height; iy++)
        {
            const float y = (height - 1 - iy) * invS + triangleLookup.minY;
            uint8_t * line = result.ptr<uint8_t>(iy);
            for (int ix = 0; ix < width; ix++)
            {
                const float x = ix * invS + triangleLookup.minX;
                
                const auto t = triangleLookup.lookup(x, y, &lastTriangleId);
                if (!t)
                {
                    continue;
                }

                pressureMesh.getElement(t->triangleId, pressureIds.data(), 0);
                for (int i = 0; i < elSizeP; i++)
                {
                    const int j = pressureIds[i];
                    localPressure[i] = pressure[j];
                }

                const float pVal = pressureValue(t->localX, t->localY, localPressure.data());
                const auto color = pressureScale(pVal);
               
                uint8_t * pix = line + 3 * ix;
                pix[0] = color[0];
                pix[1] = color[1];
                pix[2] = color[2];
            }
        }

        // Draw velocities
        auto velocityValue = el::getValueFunction(velocityMesh.baseElement.type);
        assert(velocityValue);
        const float maxX = triangleLookup.minX + triangleLookup.width;
        const float maxY = triangleLookup.minY + triangleLookup.height;

        auto toImg = [&](const float x, const float y) -> cv::Point2i
        {
            cv::Point2i result;
            result.x = (x - triangleLookup.minX) * imgScale + 0.5;
            result.y = height - 1 - ((y - triangleLookup.minY) * imgScale + 0.5);
            return result;
        };

        const cv::Scalar velocityColor(255, 255, 255);
        for (float y = triangleLookup.minY + velocityStep / 2; y <= maxY - velocityStep / 2; y += velocityStep)
        {
            for (float x = triangleLookup.minX + velocityStep / 2; x <= maxX - velocityStep / 2; x += velocityStep)
            {
                const auto t = triangleLookup.lookup(x, y, &lastTriangleId);
                if (!t)
                {
                    continue;
                }

                velocityMesh.getElement(t->triangleId, velocityIds.data(), 0);
                for (int i = 0; i < elSizeV; i++)
                {
                    const int j = velocityIds[i];
                    localVx[i] = vx[j];
                    localVy[i] = vy[j];
                }

                const float dx = velocityValue(t->localX, t->localY, localVx.data());
                const float dy = velocityValue(t->localX, t->localY, localVy.data());
                
                const float tx = x + velocityScale * dx;
                const float ty = y + velocityScale * dy;

                const auto origin = toImg(x, y);
                const auto dst = toImg(tx, ty);

                // cv::Vec3b localColor = result.at<cv::Vec3b>(origin.y, origin.x);
                // const cv::Scalar invertedColor(255 - localColor[0], 255 - localColor[1], 255 - localColor[2]);

                cv::circle(result, origin, 1, velocityColor, cv::FILLED);
                cv::line(result, origin, dst, velocityColor);
            }
        }

        return result;
    }
}