#include <mesh/drawMesh.h>

#include <limits>

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
}