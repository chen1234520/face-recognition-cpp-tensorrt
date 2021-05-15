#ifndef RETINA_FACE_H
#define RETINA_FACE_H

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "utils.h"

using namespace nvinfer1;

#define CLIP(a, min, max) (MAX(MIN(a, max), min)) // MIN, MAX defined in opencv

struct anchorBox {
    float cx;
    float cy;
    float sx;
    float sy;
};

class RetinaFace {
  public:
    RetinaFace(Logger gLogger, const std::string engineFile, int frameWidth, int frameHeight,
               std::vector<int> inputShape, int maxFacesPerScene, float nms_threshold, float bbox_threshold);
    ~RetinaFace();
    std::vector<struct Bbox> findFace(cv::Mat &img);

  private:
    int m_frameWidth, m_frameHeight;        //视频分辨率尺寸
    int m_INPUT_C, m_INPUT_H, m_INPUT_W;    //网络输入输入尺寸
    int m_OUTPUT_SIZE_BASE, m_maxFacesPerScene;
    float m_nms_threshold, m_bbox_threshold;
    static const int m_batchSize = 1;
    float m_scale_h;    //高度缩放比例(视频帧尺寸相对于网络输入尺寸)
    float m_scale_w;
    cv::Mat m_input;
    float *m_output0, *m_output1;
    std::vector<struct Bbox> m_outputBbox;

    Logger m_gLogger;
    std::string m_engineFile;
    DataType m_dtype;
    ICudaEngine *m_engine;
    IExecutionContext *m_context;
    cudaStream_t stream;
    void *buffers[3];
    int inputIndex, outputIndex0, outputIndex1;

    void loadEngine(Logger gLogger, const std::string engineFile);
    void preInference();
    void doInference(float *input, float *output0, float *output1);
    void preprocess(cv::Mat &img);
    void postprocessing(float *bbox, float *conf);
    void create_anchor_retinaface(std::vector<anchorBox> &anchor, int w, int h);
    static inline bool m_cmp(Bbox a, Bbox b);
    void nms(std::vector<Bbox> &input_boxes, float NMS_THRESH);
};

#endif
