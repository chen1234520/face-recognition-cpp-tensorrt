#ifndef ARCFACE_IR50_H
#define ARCFACE_IR50_H

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include <NvInferPlugin.h>
#include <map>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "utils.h"

using namespace nvinfer1;

class ArcFaceIR50 {
  public:
    ArcFaceIR50(Logger gLogger, const std::string engineFile, int frameWidth, int frameHeight,
                std::vector<int> inputShape, int outputDim, int maxFacesPerScene, float knownPersonThreshold);
    ~ArcFaceIR50();

    void preprocessFace(cv::Mat &face, cv::Mat &output);
    void preprocessFaces();
    void preprocessFaces_();
    void doInference(float *input, float *output);
    void doInference(float *input, float *output, int batchSize);
    void forwardAddFace(cv::Mat image, std::vector<struct Bbox> outputBbox, const std::string className);
    void addEmbedding(const std::string className, std::vector<float> embedding);
    void forward(cv::Mat image, std::vector<struct Bbox> outputBbox);
    float *featureMatching();
    std::tuple<std::vector<std::string>, std::vector<float>> getOutputs(float *output_sims);
    void visualize(cv::Mat &image, std::vector<std::string> names, std::vector<float> sims);
    void addNewFace(cv::Mat &image, std::vector<struct Bbox> outputBbox);
    void resetVariables();
    void initKnownEmbeds(int num);
    void initCosSim();

    std::vector<struct CroppedFace> croppedFaces;   //检测到的所有人脸
    std::vector<struct KnownID> knownFaces;    //存储人脸库中的所有人脸ID
    static int classCount;  //人脸库中类别数

  private:
    const char *m_INPUT_BLOB_NAME = "input";
    const char *m_OUTPUT_BLOB_NAME = "output";
    int m_frameWidth, m_frameHeight;
    int m_INPUT_C, m_INPUT_H, m_INPUT_W;    //网络输入尺寸
    int m_OUTPUT_D, m_OUTPUT_SIZE_BASE;     //m_OUTPUT_D:网络输出维度
    int m_INPUT_SIZE, m_OUTPUT_SIZE;
    int m_maxFacesPerScene;
    cv::Mat m_input;
    float *m_embed, *m_embeds;  //m_embedd:单张人脸的网络输出结果,m_embeds:所有人脸的网络输出结果
    float *m_knownEmbeds;    //存储人脸库中人脸底库的特征
    float *m_outputs;        //m_outputs:存储人脸匹配结果.(人脸库中每个类别的相似度得分.score1,score1,......score10)
    std::vector<std::vector<float>> m_embeddings;

    Logger m_gLogger;
    std::string m_engineFile;
    float m_knownPersonThresh;
    ICudaEngine *m_engine;
    IExecutionContext *m_context;
    cudaStream_t stream;
    void *buffers[2];
    int inputIndex, outputIndex;

    CosineSimilarityCalculator cossim;

    void loadEngine(Logger gLogger, const std::string engineFile);
    void preInference();
};

#endif // ARCFACE_IR50_H
