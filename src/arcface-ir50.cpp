#include "arcface-ir50.h"

int ArcFaceIR50::classCount = 0;

ArcFaceIR50::ArcFaceIR50(Logger gLogger, const std::string engineFile, int frameWidth, int frameHeight,
                         std::vector<int> inputShape, int outputDim, int maxFacesPerScene, float knownPersonThreshold) {
    m_frameWidth = static_cast<const int>(frameWidth);
    m_frameHeight = static_cast<const int>(frameHeight);
    assert(inputShape.size() == 3);
    m_INPUT_C = static_cast<const int>(inputShape[0]);
    m_INPUT_H = static_cast<const int>(inputShape[1]);
    m_INPUT_W = static_cast<const int>(inputShape[2]);
    m_OUTPUT_D = static_cast<const int>(outputDim);
    m_INPUT_SIZE = static_cast<const int>(m_INPUT_C * m_INPUT_H * m_INPUT_W * sizeof(float));
    m_OUTPUT_SIZE = static_cast<const int>(m_OUTPUT_D * sizeof(float));
    m_embed = new float[m_OUTPUT_D];
    croppedFaces.reserve(maxFacesPerScene);
    m_embeds = new float[maxFacesPerScene * m_OUTPUT_D];
    m_knownPersonThresh = knownPersonThreshold;

    // load engine from .engine file or create new engine.
    ///==从.engine文件加载引擎或创建新引擎
    loadEngine(gLogger, engineFile);

    // create stream and pre-allocate GPU buffers memory.
    ///==创建流并预分配GPU缓冲区内存
    preInference();
}

void ArcFaceIR50::loadEngine(Logger gLogger, const std::string engineFile) {
    if (fileExists(engineFile)) {
        std::cout << "[INFO] Loading ArcFace Engine...\n";
        std::vector<char> trtModelStream_;
        size_t size{0};

        std::ifstream file(engineFile, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);    //从打开的文件流中获取信息
            size = file.tellg();
            file.seekg(0, file.beg);    //从打开的文件流中获取信息
            trtModelStream_.resize(size);
            file.read(trtModelStream_.data(), size);
            file.close();
        }

        ///==step1.通过RunTime对象反序列化读取engine引擎
        IRuntime *runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        m_engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
        assert(m_engine != nullptr);

        ///==step2.创建context来开辟空间存储中间值，一个engine可以有多个context来并行处理
        m_context = m_engine->createExecutionContext();
        assert(m_context != nullptr);
    } else {
        throw std::logic_error("Cant find engine file");
    }
}

void ArcFaceIR50::preInference() {
    /*
     * function:推理流程
     */

    // Engine requires exactly IEngine::getNbBindings() number of buffers.引擎完全需要IEngine::getNbBindings()缓冲区数。
    assert(m_engine->getNbBindings() == 2);

    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than IEngine::getNbBindings()
    // 为了绑定缓冲区，我们需要知道输入和输出张量的名称。 请注意，索引必须小于IEngine::getNbBindings()
    ///==step1.指定输入和输出节点名来获取输入输出索引
    inputIndex = m_engine->getBindingIndex(m_INPUT_BLOB_NAME);
    outputIndex = m_engine->getBindingIndex(m_OUTPUT_BLOB_NAME);

    // Create GPU buffers on device.
    ///==step2.在设备上创建GPU缓冲区
    checkCudaStatus(cudaMalloc(&buffers[inputIndex], m_INPUT_SIZE));
    checkCudaStatus(cudaMalloc(&buffers[outputIndex], m_OUTPUT_SIZE));

    // Create stream.
    ///==step3.创建流
    checkCudaStatus(cudaStreamCreate(&stream));
}

void ArcFaceIR50::preprocessFace(cv::Mat &face, cv::Mat &output) {
    cv::cvtColor(face, face, cv::COLOR_BGR2RGB);
    face.convertTo(face, CV_32F);
    face = (face - cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125;
    std::vector<cv::Mat> temp;
    cv::split(face, temp);
    for (int i = 0; i < temp.size(); i++) {
        output.push_back(temp[i]);
    }
}

void ArcFaceIR50::preprocessFaces() {
    /*
     * function:前处理操作.(数据类型转换、减均值、归一化)
     */
    for (int i = 0; i < croppedFaces.size(); i++) {
        cv::cvtColor(croppedFaces[i].faceMat, croppedFaces[i].faceMat, cv::COLOR_BGR2RGB);
        croppedFaces[i].faceMat.convertTo(croppedFaces[i].faceMat, CV_32F);
        croppedFaces[i].faceMat = (croppedFaces[i].faceMat - cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125;
        std::vector<cv::Mat> temp;
        cv::split(croppedFaces[i].faceMat, temp);
        for (int i = 0; i < temp.size(); i++) {
            m_input.push_back(temp[i]);
        }
        croppedFaces[i].faceMat = m_input.clone();
        m_input.release();
    }
}

void ArcFaceIR50::preprocessFaces_() {
    float *m_input_;
    m_input_ = new float[croppedFaces.size() * m_INPUT_C * m_INPUT_H * m_INPUT_W];
    for (int i = 0; i < croppedFaces.size(); i++) {
        cv::cvtColor(croppedFaces[i].faceMat, croppedFaces[i].faceMat, cv::COLOR_BGR2RGB);
        croppedFaces[i].faceMat.convertTo(croppedFaces[i].faceMat, CV_32F);
        croppedFaces[i].faceMat = (croppedFaces[i].faceMat - cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125;
        std::vector<cv::Mat> temp;
        cv::split(croppedFaces[i].faceMat, temp);
        for (int j = 0; j < temp.size(); j++) {
            m_input.push_back(temp[j]);
            std::copy(temp[j].ptr<float>(0), temp[j].ptr<float>(0) + m_INPUT_H * m_INPUT_W,
                      m_input_ + i * m_INPUT_C * m_INPUT_H * m_INPUT_W + j * m_INPUT_H * m_INPUT_W);
        }
        croppedFaces[i].faceMat = m_input.clone();
        m_input.release();
    }
}

void ArcFaceIR50::doInference(float *input, float *output) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    // DMA将批处理数据输入到设备，异步推断该批处理，并将DMA输出返回给主机
    ///=拷贝数据 从主机--->设备
    checkCudaStatus(cudaMemcpyAsync(buffers[inputIndex], input, m_INPUT_SIZE, cudaMemcpyHostToDevice, stream));

    ///=异步执行推理
    m_context->enqueueV2(buffers, stream, nullptr);

    ///=拷贝数据 从设备--->主机
    checkCudaStatus(cudaMemcpyAsync(output, buffers[outputIndex], m_OUTPUT_SIZE, cudaMemcpyDeviceToHost, stream));

    ///=同步流
    cudaStreamSynchronize(stream);  //因为上面的cudaMemcpyAsync()函数是异步方式执行的,所有这里需要进行同步
}

void ArcFaceIR50::doInference(float *input, float *output, int batchSize) {
    // Set input dimensions
    //std::cout << "batchSize: " << batchSize << "\n";
    //m_context->setOptimizationProfile(batchSize - 1);
    m_context->setBindingDimensions(inputIndex, Dims4(batchSize, m_INPUT_C, m_INPUT_H, m_INPUT_W));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    checkCudaStatus(
        cudaMemcpyAsync(buffers[inputIndex], input, batchSize * m_INPUT_SIZE, cudaMemcpyHostToDevice, stream));
    //m_context->enqueue(batchSize, buffers, stream, nullptr);
    m_context->enqueueV2(buffers, stream, nullptr);
    checkCudaStatus(
        cudaMemcpyAsync(output, buffers[outputIndex], batchSize * m_OUTPUT_SIZE, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void ArcFaceIR50::forwardAddFace(cv::Mat image, std::vector<struct Bbox> outputBbox, const std::string className) {
    ///==裁剪出人脸图像
    getCroppedFaces(image, outputBbox, m_INPUT_W, m_INPUT_H, croppedFaces);
    if (!croppedFaces.empty()) {
        ///==前处理(减均值、归一化等)
        preprocessFaces();

        ///==前向推理
        doInference((float *)croppedFaces[0].faceMat.ptr<float>(0), m_embed);
        struct KnownID person;
        person.className = className;
        person.embeddedFace.insert(person.embeddedFace.begin(), m_embed, m_embed + m_OUTPUT_D);
        knownFaces.push_back(person);
        classCount++;
    }
    croppedFaces.clear();
}

void ArcFaceIR50::addEmbedding(const std::string className, std::vector<float> embedding) {
    struct KnownID person;
    person.className = className;
    //person.embeddedFace = embedding;
    knownFaces.push_back(person);
    std::copy(embedding.begin(), embedding.end(), m_knownEmbeds + classCount * m_OUTPUT_D); //将人脸特征复制到m_knownEmbeds中
    classCount++;
}

void ArcFaceIR50::initKnownEmbeds(int num) { m_knownEmbeds = new float[num * m_OUTPUT_D]; }

void ArcFaceIR50::initCosSim() { cossim.init(m_knownEmbeds, classCount, m_OUTPUT_D); }

void ArcFaceIR50::forward(cv::Mat frame, std::vector<struct Bbox> outputBbox) {
    ///==裁剪出人脸图像
    getCroppedFaces(frame, outputBbox, m_INPUT_W, m_INPUT_H, croppedFaces);

    ///==前处理(减均值、归一化等)
    preprocessFaces();
    for (int i = 0; i < croppedFaces.size(); i++) {
        ///==前向推理
        doInference((float *)croppedFaces[i].faceMat.ptr<float>(0), m_embed);

        //拷贝函数.将序列A的内容拷贝到序列B中.参数1:序列A的首地址,参数2:序列A的尾地址+1,参数3:序列B的首地址
        std::copy(m_embed, m_embed + m_OUTPUT_D, m_embeds + i * m_OUTPUT_D);
    }
}

float *ArcFaceIR50::featureMatching() {
    m_outputs = new float[croppedFaces.size() * classCount];
    //float *m_outputs_ = new float[croppedFaces.size() * classCount];
    if (knownFaces.size() > 0 && croppedFaces.size() > 0) {
        //auto start = std::chrono::high_resolution_clock::now();
        //batch_cosine_similarity(m_embeds, m_knownEmbeds, croppedFaces.size(), classCount, m_OUTPUT_D, m_outputs_);
        //auto end = std::chrono::high_resolution_clock::now();
        //std::cout << "\tOpenBLAS: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000. << "ms\n";
        cossim.calculate(m_embeds, croppedFaces.size(), m_outputs);
        // assertion
        //for (int i = 0; i < croppedFaces.size(); ++i) {
        //for (int j = 0; j < classCount; ++j) {
        //std::cout << *(m_outputs + i * classCount + j) << " " << *(m_outputs_ + j * croppedFaces.size() + i) << "\n";
        //std::cout << *(m_outputs + i * classCount + j) << " " << *(m_outputs_ + i * classCount + j) << "\n";
        //std::cout << "=================\n";
        //assert(fabs(*(m_outputs + i * classCount + j) - *(m_outputs_ + i * classCount + j)) <= 0.000001);
        //}
        //}
    } else {
        std::cout << "No faces in database or no faces found\n";
    }
    return m_outputs;
}

std::tuple<std::vector<std::string>, std::vector<float>> ArcFaceIR50::getOutputs(float *output_sims) {
    std::vector<std::string> names;
    std::vector<float> sims;
    ///===for循环所有人脸的匹配结果,并取出人脸的ID和score保存
    for (int i = 0; i < croppedFaces.size(); ++i) {

        for(int k = i * classCount; k < (i + 1) * classCount; k++){
            std::cout << "\t " << *(output_sims + k);
        }
        std::cout<< std::endl;

        int argmax = std::distance(output_sims + i * classCount,
                                   std::max_element(output_sims + i * classCount, output_sims + (i + 1) * classCount));//其实是找到得分最高的faceid
        //max_element:返回范围内最大元素的索引地址。distance:计算两个元素之间的索引距离
        float sim = *(output_sims + i * classCount + argmax);   //得到最高的得分score
        std::string name = knownFaces[argmax].className;        //得到score最大的人脸id
        names.push_back(name);  //name:人脸ID
        sims.push_back(sim);    //sim:得分
    }
    return std::make_tuple(names, sims);
}

void ArcFaceIR50::visualize(cv::Mat &image, std::vector<std::string> names, std::vector<float> sims) {
    for (int i = 0; i < croppedFaces.size(); ++i) {
        std::cout << "faceID:" << names[i] << "\tscore:" << sims[i] << std::endl;

        float fontScaler = static_cast<float>(croppedFaces[i].x2 - croppedFaces[i].x1) / static_cast<float>(m_frameWidth);
        cv::Scalar color;
        if (sims[i] >= m_knownPersonThresh)
            color = cv::Scalar(0, 255, 0);
        else
            color = cv::Scalar(0, 0, 255);
        cv::rectangle(image, cv::Point(croppedFaces[i].y1, croppedFaces[i].x1),
                      cv::Point(croppedFaces[i].y2, croppedFaces[i].x2), color, 2, 8, 0);
        cv::putText(image, names[i] + " " + std::to_string(sims[i]),
                    cv::Point(croppedFaces[i].y1 + 2, croppedFaces[i].x2 - 3), cv::FONT_HERSHEY_DUPLEX,
                    0.1 + 2 * fontScaler, color, 1);
    }
}

void ArcFaceIR50::addNewFace(cv::Mat &image, std::vector<struct Bbox> outputBbox) {
    std::cout << "Adding new person...\nPlease make sure there is only one "
                 "face in the current frame.\n"
              << "What's your name? ";
    std::string newName;
    std::cin >> newName;
    std::cout << "Hi " << newName << ", you will be added to the database.\n";
    forwardAddFace(image, outputBbox, newName);
    //std::string filePath = "../imgs/";
    //filePath.append(newName);
    //filePath.append(".jpg");
    //cv::imwrite(filePath, image);
}

void ArcFaceIR50::resetVariables() {
    //m_embeddings.clear();
    croppedFaces.clear();
}

ArcFaceIR50::~ArcFaceIR50() {
    // Release stream and buffers
    checkCudaStatus(cudaStreamDestroy(stream));
    checkCudaStatus(cudaFree(buffers[inputIndex]));
    checkCudaStatus(cudaFree(buffers[outputIndex]));
}
