#include "utils.h"

void getFilePaths(std::string rootPath, std::vector<struct Paths> &paths) {
    /*
    imagesPath--|
                |--class0--|
                |          |--f0.jpg
                |          |--f1.jpg
                |
                |--class1--|
                           |--f0.jpg
                           |--f1.jpg
    ...
    */
    DIR *dir;
    struct dirent *entry;
    std::string postfix = ".jpg";
    if ((dir = opendir(rootPath.c_str())) != NULL) {
        while ((entry = readdir(dir)) != NULL) {
            std::string class_path = rootPath + "/" + entry->d_name;
            DIR *class_dir = opendir(class_path.c_str());
            struct dirent *file_entry;
            while ((file_entry = readdir(class_dir)) != NULL) {
                std::string name(file_entry->d_name);
                if (name.length() >= postfix.length() &&
                    0 == name.compare(name.length() - postfix.length(), postfix.length(), postfix))
                    if (file_entry->d_type != DT_DIR) {
                        struct Paths tempPaths;
                        tempPaths.className = std::string(entry->d_name);
                        tempPaths.absPath = class_path + "/" + name;
                        paths.push_back(tempPaths);
                    }
            }
            //closedir(class_dir);
        }
        closedir(dir);
    }
}

bool fileExists(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

// void l2_norm(float *p, int size) {	//blasint这个函数我这边没编译过,这几个相关的函数没有使用，就先暂时注释掉了
//     float norm = cblas_snrm2((blasint)size, p, 1);
//     cblas_sscal((blasint)size, 1 / norm, p, 1);
// }

void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("CUDA API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("CUDA API failed");
    }
}

void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS API failed with status " << status << "\n";
        throw std::logic_error("cuBLAS API failed");
    }
}

CosineSimilarityCalculator::CosineSimilarityCalculator() {
    checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
    checkCublasStatus(cublasLtCreate(&ltHandle));
    checkCudaStatus(cudaStreamCreate(&stream));
}

void CosineSimilarityCalculator::init(float *knownEmbeds, int numRow, int numCol) {
    /*
    Calculate C = A x B
    Input:
        A: m x k, row-major matrix 行主矩阵
        B: n x k, row-major matrix
    Output:
        C: m x n, row-major matrix

    NOTE: Since cuBLAS use column-major matrix as input, we need to transpose A (transA=CUBLAS_OP_T).
    注意：由于cuBLAS使用列主矩阵作为输入，因此我们需要转置A（transA = CUBLAS_OP_T）。
    */
    m = static_cast<const int>(numRow);
    k = static_cast<const int>(numCol);
    lda = static_cast<const int>(numCol);
    ldb = static_cast<const int>(numCol);
    ldc = static_cast<const int>(numRow);

    // alloc and copy known embeddings to GPU
    /// 分配内存,并将人脸底库的嵌入复制到GPU
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dA), m * k * sizeof(float)));
    checkCudaStatus(cudaMemcpyAsync(dA, knownEmbeds, m * k * sizeof(float), cudaMemcpyHostToDevice, stream));

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults;
    // here we just need to set the transforms for A and B
    // 创建操作描述符 有关默认值的详细信息，请参见cublasLtMatmulDescAttributes_t。
    // 在这里，我们只需要设置A和B的转换

    ///==cublas矩阵乘法实现步骤
    ///==Step1.创建矩阵乘法的描述符
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, dataType));//cublasLtMatmulDescCreate:该函数通过分配所需的内存来创建矩阵乘法描述符。

    ///==step2.为描述符指定相关的值(应该是指定需要做乘法的矩阵的数据)
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));//此函数为前面创建的矩阵乘法描述符指定属性的值。
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    // 创建矩阵描述符，我们对这里的细节很满意，因此无需设置任何额外的属性

    ///==创建矩阵乘法的描述符
    //cublasLtMatrixLayoutCreate:该函数通过分配所需的内存来创建矩阵乘法描述符。
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, dataType, transa == CUBLAS_OP_N ? m : k,
                                                 transa == CUBLAS_OP_N ? k : m, lda));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    // 创建首选项句柄；我们可以使用额外的属性来禁用张量运算，或确保选择的算法将与错误对齐的A，B，C一起使用；
    // 在这里为简单起见，我们假设A，B，C始终对齐（例如，直接来自cudaMalloc）

    // cublasLtMatmulPreferenceCreate:该函数通过分配所需的内存来创建矩阵乘法启发式搜索首选项描述符。
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));

    // cublasLtMatmulPreferenceSetAttribute:此函数为前面创建的矩阵乘法首选项描述符设置指定属性的值。
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &workspaceSize, sizeof(workspaceSize))); //
}

void CosineSimilarityCalculator::calculate(float *embeds, int embedCount, float *outputs) {
    n = embedCount;

    // Allocate arrays on GPU
    ///==step1.cudaMalloc:在GPU上为矩阵申请一块内存.并将数据从主机拷到设备中
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dB), k * n * sizeof(float)));     //人脸特征维度*检测到人脸数量
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dC), m * n * sizeof(float)));     //人脸库中人脸数量*检测到人脸数量
    checkCudaStatus(cudaMemcpyAsync(dB, embeds, k * n * sizeof(float), cudaMemcpyHostToDevice, stream)); //cudaMemcpyAsync:主机和设备之间数据传递

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    // 创建矩阵描述符，我们在这里的细节很好，所以不需要设置任何额外的属性

    ///==step2.cublasLtMatrixLayoutCreate:创建矩阵布局描述符
    cublasLtMatrixLayout_t Bdesc = NULL, Cdesc = NULL;
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, dataType, transb == CUBLAS_OP_N ? k : n,
                                                 transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, dataType, m, n, ldc));     //cublasLtMatrixLayoutCreate:创建矩阵布局描述符

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    // 我们只需要最好的启发式来尝试运行matmul。无法保证这会起作用，
    // 例如，如果A对齐不好，您可以请求更多（例如32）个算法，并尝试逐个运行它们，直到某个算法起作用为止
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    //cublasLtMatmulAlgoGetHeuristic:检索矩阵乘法的最佳算法,结果返回在最后两个参数中(参数9：以递增的方式存放每个算法时间;参数10:返回参数9中的算法数量)
    ///==step3.检索矩阵乘法的最佳算法
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1,
                                                     &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    // Do the actual multiplication.做实际的乘法.cublasLtMatmul:矩阵乘法
    ///==step4.执行矩阵乘法
    checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Cdesc, dC, Cdesc,
                                     &heuristicResult.algo, workspace, workspaceSize, stream));

    // Cleanup: descriptors are no longer needed as all GPU work was already enqueued
    // 清理：由于所有GPU工作都已入队，因此不再需要描述符
    ///==step5.销毁之前创建的矩阵描述符
    if (Cdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));  //cublasLtMatrixLayoutDestroy:此函数将销毁先前创建的矩阵布局描述符对象。
    if (Bdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));

    // Copy the result on host memory.//将结果复制到主机内存中

    ///==step6.数据从设备拷贝回主机
    checkCudaStatus(cudaMemcpyAsync(outputs, dC, m * n * sizeof(float), cudaMemcpyDeviceToHost, stream));//cudaMemcpyAsync:主机和设备之间数据传递

    // CUDA stream sync.CUDA流同步
    ///==step7.执行CUDA流同步(因为前面的拷贝是异步操作的)
    checkCudaStatus(cudaStreamSynchronize(stream));

    // Free GPU memory.释放GPU内存
    checkCudaStatus(cudaFree(dB));
    checkCudaStatus(cudaFree(dC));
}

CosineSimilarityCalculator::~CosineSimilarityCalculator() {
    if (preference)
        checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Adesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc)
        checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));

    checkCublasStatus(cublasLtDestroy(ltHandle));
    checkCudaStatus(cudaFree(dA));
    checkCudaStatus(cudaFree(workspace));
    checkCudaStatus(cudaStreamDestroy(stream));
}

void cublas_batch_cosine_similarity(float *A, float *B, int m, int n, int k, float *outputs) {
    const int lda = k, ldb = k, ldc = m;
    const float alpha = 1, beta = 0;

    cublasLtHandle_t ltHandle;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    void *workspace;
    size_t workspaceSize = 1024 * 1024 * 4;
    cudaStream_t stream;
    checkCublasStatus(cublasLtCreate(&ltHandle));
    checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
    checkCudaStatus(cudaStreamCreate(&stream));

    // Allocate arrays on GPU
    auto start = std::chrono::high_resolution_clock::now();
    float *dA, *dB, *dC;
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dA), m * k * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dB), k * n * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&dC), m * n * sizeof(float)));

    checkCudaStatus(cudaMemcpyAsync(dA, A, m * k * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(dB, B, k * n * sizeof(float), cudaMemcpyHostToDevice, stream));

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\tAllo & cpy: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms\n";

    start = std::chrono::high_resolution_clock::now();
    // cuBLASLt init
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUDA_R_32F));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k,
                                                 transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n,
                                                 transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1,
                                                     &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    // Do the actual multiplication
    checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Cdesc, dC, Cdesc,
                                     &heuristicResult.algo, workspace, workspaceSize, 0));

    // Cleanup: descriptors are no longer needed as all GPU work was already enqueued
    if (preference)
        checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc)
        checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    end = std::chrono::high_resolution_clock::now();
    std::cout << "\tMatmul & cleanup: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms\n";

    start = std::chrono::high_resolution_clock::now();
    // Copy the result on host memory
    checkCudaStatus(cudaMemcpyAsync(outputs, dC, m * n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    checkCudaStatus(cudaStreamSynchronize(stream));

    // Free GPU memory
    checkCublasStatus(cublasLtDestroy(ltHandle));
    checkCudaStatus(cudaFree(dA));
    checkCudaStatus(cudaFree(dB));
    checkCudaStatus(cudaFree(dC));
    checkCudaStatus(cudaFree(workspace));
    checkCudaStatus(cudaStreamDestroy(stream));
    end = std::chrono::high_resolution_clock::now();
    std::cout << "\tCpy & free: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms\n";
}

// void batch_cosine_similarity(float *A, float *B, int embedCount, int classCount, int size, float *outputs) {
//     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (blasint)embedCount, (blasint)classCount, size, 1, A, size, B,
//                 size, 0, outputs, classCount);
// }

// std::vector<std::vector<float>> batch_cosine_similarity(std::vector<std::vector<float>> &A,
//                                                         std::vector<struct KnownID> &B, const int size,
//                                                         bool normalize) {
//     std::vector<std::vector<float>> outputs;
//     if (normalize) {
//         // Calculate cosine similarity
//         for (int A_index = 0; A_index < A.size(); ++A_index) {
//             std::vector<float> output;
//             for (int B_index = 0; B_index < B.size(); ++B_index) {
//                 float *p_A = &A[A_index][0];
//                 float *p_B = &B[B_index].embeddedFace[0];
//                 float sim = cblas_sdot((blasint)size, p_A, 1, p_B, 1);
//                 output.push_back(sim);
//             }
//             outputs.push_back(output);
//         }
//     } else {
//         // Pre-calculate norm for all elements
//         std::vector<float> A_norms, B_norms;
//         for (int i = 0; i < A.size(); ++i) {
//             float *p = &A[i][0];
//             float norm = cblas_snrm2((blasint)size, p, 1);
//             A_norms.push_back(norm);
//         }
//         for (int i = 0; i < B.size(); ++i) {
//             float *p = &B[i].embeddedFace[0];
//             float norm = cblas_snrm2((blasint)size, p, 1);
//             B_norms.push_back(norm);
//         }
//         // Calculate cosine similarity
//         for (int A_index = 0; A_index < A.size(); ++A_index) {
//             std::vector<float> output;
//             for (int B_index = 0; B_index < B.size(); ++B_index) {
//                 float *p_A = &A[A_index][0];
//                 float *p_B = &B[B_index].embeddedFace[0];
//                 float sim = cblas_sdot((blasint)size, p_A, 1, p_B, 1) / (A_norms[A_index] * B_norms[B_index]);
//                 output.push_back(sim);
//             }
//             outputs.push_back(output);
//         }
//     }
//     return outputs;
// }

// float cosine_similarity(std::vector<float> &A, std::vector<float> &B) {
//     if (A.size() != B.size()) {
//         std::cout << A.size() << " " << B.size() << std::endl;
//         throw std::logic_error("Vector A and Vector B are not the same size");
//     }
//
//     // Prevent Division by zero
//     if (A.size() < 1) {
//         throw std::logic_error("Vector A and Vector B are empty");
//     }
//
//     float *p_A = &A[0];
//     float *p_B = &B[0];
//     float mul = cblas_sdot((blasint)(A.size()), p_A, 1, p_B, 1);
//     float d_a = cblas_sdot((blasint)(A.size()), p_A, 1, p_A, 1);
//     float d_b = cblas_sdot((blasint)(A.size()), p_B, 1, p_B, 1);
//
//     if (d_a == 0.0f || d_b == 0.0f) {
//         throw std::logic_error("cosine similarity is not defined whenever one or both "
//                                "input vectors are zero-vectors.");
//     }
//
//     return mul / (sqrt(d_a) * sqrt(d_b));
// }

void getCroppedFaces(cv::Mat frame, std::vector<struct Bbox> &outputBbox, int resize_w, int resize_h,
                     std::vector<struct CroppedFace> &croppedFaces) {
    for (std::vector<struct Bbox>::iterator it = outputBbox.begin(); it != outputBbox.end(); it++) {
        cv::Rect facePos(cv::Point((*it).y1, (*it).x1), cv::Point((*it).y2, (*it).x2));
        cv::Mat tempCrop = frame(facePos);
        struct CroppedFace currFace;
        cv::resize(tempCrop, currFace.faceMat, cv::Size(resize_h, resize_w), 0, 0, cv::INTER_CUBIC);
        currFace.face = currFace.faceMat.clone();
        currFace.x1 = it->x1;
        currFace.y1 = it->y1;
        currFace.x2 = it->x2;
        currFace.y2 = it->y2;
        croppedFaces.push_back(currFace);
    }
}

Requests::Requests(std::string server, int location) {
    m_headers = curl_slist_append(m_headers, "Content-Type: application/json");
    m_curl = curl_easy_init();
    curl_easy_setopt(m_curl, CURLOPT_URL, server.c_str());
    curl_easy_setopt(m_curl, CURLOPT_HTTPHEADER, m_headers);
    curl_easy_setopt(m_curl, CURLOPT_CUSTOMREQUEST, "POST");

    m_location = std::to_string(location);
}

void Requests::send(std::vector<std::string> names, std::vector<float> sims,
                    std::vector<struct CroppedFace> &croppedFaces, int classCount, float threshold,
                    std::string check_type) {
    std::vector<json> data;
    for (int i = 0; i < croppedFaces.size(); ++i) {
        std::cout << names[i] << " " << sims[i] << "\n";
        if (sims[i] < threshold)
            continue;

        // cv::Mat to base64
        std::vector<uchar> buf;
        cv::imencode(".jpg", croppedFaces[i].face, buf);
        auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
        std::string encoded = base64_encode(enc_msg, buf.size());

        // create json element
        json d = {
            {"image", encoded},
            {"userId", names[i]},
            {"conf", sims[i]},
            {"type", check_type},
        };
        data.push_back(d);
    }
    if (data.size() < 1)
        return;

    // payload prepare
    json info_detection = {
        {"location", m_location},
        {"array", data},
    };
    std::string payload = info_detection.dump();
    // std::cout << payload.c_str() << "\n";

    // Send
    curl_easy_setopt(m_curl, CURLOPT_POSTFIELDS, payload.c_str());
    res = curl_easy_perform(m_curl);
}

Requests::~Requests() {
    // Clean up
    curl_easy_cleanup(m_curl);
    m_curl = NULL;
    curl_slist_free_all(m_headers);
    m_headers = NULL;
}
