#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <opencv2/highgui.hpp>
#include <string>

#include "arcface-ir50.h"
#include "json.hpp"
#include "retinaface.h"
#include "utils.h"

using json = nlohmann::json;
#define LOG_TIMES

int main(/*int argc, const char **argv*/) {
    // Config
    std::cout << "[INFO] Loading config..." << std::endl;
    std::string configPath = "../config.json";
    // if (argc < 2 || (strcmp(argv[1], "-c") != 0)) {
    //     std::cout << "\tPlease specify config file path with -c option. Use default path: \"" << configPath << "\"\n";
    // } else {
    //     configPath = argv[2];
    //     std::cout << "\tConfig path: \"" << configPath << "\"\n";
    // }
    std::ifstream configStream(configPath);
    json config;
    configStream >> config;
    configStream.close();

    // TRT Logger
    Logger gLogger = Logger();

    // curl request
    Requests r(config["send_server"], config["send_location"]);

    // params
    int numFrames = 0;
    std::string detEngineFile = config["det_engine"];   //检测模型路径
    // std::vector<int> detInputShape = config["det_inputShape"];  //检测模型输入尺寸?? [3, 288, 320]
    std::vector<int> detInputShape = {3, 288, 320};  //检测模型输入尺寸?? [3, 288, 320]
    float det_threshold_nms = config["det_threshold_nms"];      //检测NMS阈值0.4
    float det_threshold_bbox = config["det_threshold_bbox"];    //bbox阈值??? 0.6
    std::vector<int> recInputShape = config["rec_inputShape"];  //识别模型输入尺寸[3, 112, 112]
    int recOutputDim = config["rec_outputDim"];             //人脸识别输出特征维度512
    std::string recEngineFile = config["rec_engine"];       //识别模型路径
    int videoFrameWidth = config["input_frameWidth"];       //视频分辨率宽度
    int videoFrameHeight = config["input_frameHeight"];     //视频分辨率高度
    int maxFacesPerScene = config["det_maxFacesPerScene"];  //4  每张图中最大的人脸数???
    float knownPersonThreshold = config["rec_knownPersonThreshold"];    //人脸比对阈值??? 0.65
    std::string embeddingsFile = config["input_embeddingsFile"];    //人脸底库的json(保存人脸特征)

    // init arcface
    ArcFaceIR50 recognizer(gLogger, recEngineFile, videoFrameWidth, videoFrameHeight, recInputShape, recOutputDim,
                           maxFacesPerScene, knownPersonThreshold);

    // init retinaface
    RetinaFace detector(gLogger, detEngineFile, videoFrameWidth, videoFrameHeight, detInputShape, maxFacesPerScene,
                        det_threshold_nms, det_threshold_bbox);

    // init bbox and allocate memory according to maxFacesPerScene.初始化bbox并根据maxFacesPerScene分配内存
    std::vector<struct Bbox> outputBbox;    //检测到的人脸框
    outputBbox.reserve(maxFacesPerScene);   //reserve为vector容器预留空间

    // create or get embeddings of known faces.创建或获取人脸底库的embeddings
    if (fileExists(embeddingsFile)) {   ///case1.读取人脸库json
        std::cout << "[INFO] Reading embeddings from file...\n";
        std::ifstream i(config["input_numImagesFile"]);     //底库中人脸数量
        std::string numImages_str;
        std::getline(i, numImages_str);
        unsigned int numImages = stoi(numImages_str);
        i.close();
        i.clear();
        i.open(embeddingsFile);
        json j;
        i >> j;
        i.close();
        recognizer.initKnownEmbeds(numImages);
        for (json::iterator it = j.begin(); it != j.end(); ++it)
            for (int i = 0; i < it.value().size(); ++i)
                recognizer.addEmbedding(it.key(), it.value()[i]);   //读取底库中人脸特征
        std::cout << "[INFO] Init cuBLASLt cosine similarity calculator...\n";
        recognizer.initCosSim();  //初始化cuBLASLt余弦相似度计算器
    } else {    ///case2.读取底库照片生成人脸json库
        std::cout << "[INFO] Parsing images from " << config["gen_imgSource"] << "\n";
        std::vector<struct Paths> paths;
        getFilePaths(config["gen_imgSource"], paths);
        unsigned int img_count = paths.size();
        std::ofstream o(config["input_numImagesFile"]);
        o << img_count << std::endl;
        o.close();
        o.clear();
        o.open(embeddingsFile);
        json j;
        cv::Mat image;
        if (config["gen_imgIsCropped"]) {   //底库照片中人脸是否裁剪出来了
            cv::Mat input;
            float output[recOutputDim];
            std::vector<float> embeddedFace;
            for (int i = 0; i < paths.size(); i++) {
                image = cv::imread(paths[i].absPath.c_str());
                std::string className = paths[i].className;
                recognizer.preprocessFace(image, input);    //预处理
                recognizer.doInference((float *)input.ptr<float>(0), output);   //执行推理
                embeddedFace.insert(embeddedFace.begin(), output, output + recOutputDim);   //保存人脸特征
                if (j.contains(className)) {
                    j[className].push_back(embeddedFace);
                } else {
                    std::vector<std::vector<float>> temp;
                    temp.push_back(embeddedFace);
                    j[className] = temp;
                }
                input.release();
                embeddedFace.clear();
            }
        } else {
            for (int i = 0; i < paths.size(); i++) {
                image = cv::imread(paths[i].absPath.c_str());
                cv::resize(image, image, cv::Size(videoFrameWidth, videoFrameHeight));
                outputBbox = detector.findFace(image);      //检测人脸
                std::string rawName = paths[i].className;
                recognizer.forwardAddFace(image, outputBbox, rawName);  //提取人脸特征
                recognizer.resetVariables();
            }
            // to json
            for (int k = 0; k < recognizer.knownFaces.size(); ++k) {
                std::string className = recognizer.knownFaces[k].className;
                std::vector<std::vector<float>> temp;
                temp.push_back(recognizer.knownFaces[k].embeddedFace);
                j[className] = temp;
            }
        }
        // write result to json file
        o << std::setw(4) << j << std::endl;
        std::cout << "[INFO] Embeddings saved to json. Exitting..." << std::endl;
        exit(0);
    }

    // init opencv and output vectors
    std::string camera_input = config["input_camera"];  //视频地址
    cv::VideoCapture vc(camera_input);
    if (!vc.isOpened()) {
        // error in opening the video input
        std::cerr << "Failed to open camera.\n";
        return -1;
    }
    cv::Mat rawInput;
    std::vector<int> coord = config["input_cropPos"]; // x1 y1 x2 y2 [470, 400, 1150, 900] w:680 h:500
    cv::Rect cropPos(cv::Point(coord[0], coord[1]), cv::Point(coord[2], coord[3]));
    cv::Mat frame;
    float *output_sims;   //存储人脸匹配结果.(人脸库中每个类别的相似度得分.score1,score1,......score10)
    std::vector<std::string> names;     //face id
    std::vector<float> sims;    //score

    std::cout << "[INFO] Start video stream\n";
    auto globalTimeStart = std::chrono::high_resolution_clock::now();
    // loop over frames with inference
    while (true) {
        bool ret = vc.read(rawInput);
        if (!ret) {
            std::cerr << "ERROR: Cannot read frame from stream\n";
            continue;
        }
        //std::cout << "Input: " << rawInput.size() << "\n";
        if (config["input_takeCrop"])   //false 可能是裁剪参数
            rawInput = rawInput(cropPos);
        cv::resize(rawInput, frame, cv::Size(videoFrameWidth, videoFrameHeight));

        ///===step1.detector Face
        double t1 = (double)cv::getTickCount();
        auto startDetect = std::chrono::high_resolution_clock::now();
        outputBbox = detector.findFace(frame);  //检测人脸
        auto endDetect = std::chrono::high_resolution_clock::now();
        t1 = (double)cv::getTickCount() - t1;
        std::cout << "detector time :" << t1*1000.0 / cv::getTickFrequency() << " ms \n";
        std::cout << "detector face number :" << outputBbox.size() << std::endl;

        if(outputBbox.size() > 0){
            std::cout << "" << std::endl;
        }

        ///===step2.recognizer face
        double t2 = (double)cv::getTickCount();
        auto startRecognize = std::chrono::high_resolution_clock::now();
        recognizer.forward(frame, outputBbox);  //人脸识别网络前向
        auto endRecognize = std::chrono::high_resolution_clock::now();
        t2 = (double)cv::getTickCount() - t2;
        std::cout << "recognizer time :" << t2 * 1000.0 / cv::getTickFrequency() << " ms \n";

        ///===step3.Feature matching
        auto startFeatM = std::chrono::high_resolution_clock::now();
        output_sims = recognizer.featureMatching();     //人脸匹配
        auto endFeatM = std::chrono::high_resolution_clock::now();
        std::tie(names, sims) = recognizer.getOutputs(output_sims); //获取faceid 和 得分

        // curl request
        if (config["send_request"]) {
            std::string check_type = "in";
            r.send(names, sims, recognizer.croppedFaces, recognizer.classCount, knownPersonThreshold, check_type);
        }

        // visualize.显示
        if (config["out_visualize"]) {
            recognizer.visualize(frame, names, sims);
            cv::namedWindow("frame",CV_WINDOW_NORMAL);
            cv::imshow("frame", frame);
            cv::waitKey(5);

            char keyboard = cv::waitKey(1);
            if (keyboard == 'q' || keyboard == 27)
                break;
            //else if (keyboard == 'n') {
                //auto dTimeStart = std::chrono::high_resolution_clock::now();
                //recognizer.addNewFace(frame, outputBbox);
                //auto dTimeEnd = std::chrono::high_resolution_clock::now();
                //globalTimeStart += (dTimeEnd - dTimeStart);
            //}
        }

        // clean
        recognizer.resetVariables();
        outputBbox.clear();
        names.clear();
        sims.clear();
        rawInput.release();
        frame.release();
        numFrames++;

#ifdef LOG_TIMES
        std::cout << "Detector took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(endDetect - startDetect).count() << "ms\n";
        std::cout << "Recognizer took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(endRecognize - startRecognize).count()
                  << "ms\n";
        std::cout << "Feature matching took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(endFeatM - startFeatM).count() << "ms\n";
        std::cout << "-------------------------" << std::endl;
#endif // LOG_TIMES
    }
    auto globalTimeEnd = std::chrono::high_resolution_clock::now();
    cv::destroyAllWindows();
    vc.release();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(globalTimeEnd - globalTimeStart).count();
    double seconds = double(milliseconds) / 1000.;
    double fps = numFrames / seconds;

    std::cout << "Counted " << numFrames << " frames in " << double(milliseconds) / 1000. << " seconds!"
              << " This equals " << fps << "fps.\n";

    return 0;
}
