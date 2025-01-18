#ifndef LIBS_MESH_INCLUDE_MESH_COLORSCALE
#define LIBS_MESH_INCLUDE_MESH_COLORSCALE

#include <vector>

#include <opencv2/opencv.hpp>

namespace mesh
{
    class AbstractColorScale
    {
      public:
        virtual cv::Scalar operator()(const float x) const = 0;

        virtual ~AbstractColorScale(){};
    };

    class SimpleColorScale : public AbstractColorScale
    {
        const float min, max;
        const std::vector<cv::Scalar> colors;

      public:
        SimpleColorScale(const float min, const float max, const std::vector<cv::Scalar> & colors);

        cv::Scalar operator()(const float x) const;

        ~SimpleColorScale();
    };
} // namespace mesh

#endif /* LIBS_MESH_INCLUDE_MESH_COLORSCALE */
