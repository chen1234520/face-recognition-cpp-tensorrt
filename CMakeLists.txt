cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(PROJECT_NAME face)
project(${PROJECT_NAME})# LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 11)

# OpenCV
find_package(OpenCV REQUIRED)

# find_package()命令是用来查找依赖包的，理想情况下，一句find_package()把一整个依赖包的
# 头文件包含路径、库路径、库名字、版本号等情况都获取到，后续只管用就好了。

# CUDA
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
    -O3
        -gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
)

if(CUDA_VERSION_MAJOR GREATER 9)
	message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")

	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS};
		-gencode arch=compute_72,code=sm_72
	)

endif()

# TensorRT(使用的是7.0.0.11)
message("CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} /usr/include /data_1/deep_learning_layer/TensorRT-7.0.0.11
  PATH_SUFFIXES include)	#find_path寻找包含该头文件的路径.参数HINTS:提供建议的搜索路径. PATH_SUFFIXES：指定路径的子文件夹
find_path(TENSORRT_INCLUDE_DIR NvInferPlugin.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} /data_1/deep_learning_layer/TensorRT-7.0.0.11
  PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} /data_1/deep_learning_layer/TensorRT-7.0.0.11
  PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
  find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} /data_1/deep_learning_layer/TensorRT-7.0.0.11
  PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
message(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
include(FindPackageHandleStandardArgs)
#include_directories(/data_1/deep_learning_layer/TensorRT-7.0.0.11/include)

include_directories(/data_2/Smart_City/Project/3rdParty/cuda/include)	# cuda10.1(10.0版本是不行的,有些头文件是不支持的)
link_directories(/data_2/Smart_City/Project/3rdParty/cuda/lib)
link_directories(/data_2/Smart_City/Project/3rdParty/lib)


include_directories(${TENSORRT_INCLUDE_DIR})
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()
message("TENSORRT_LIBRARY = ${TENSORRT_LIBRARY}")


link_directories(/data_2/Smart_City/Project/3rdParty/lib/)

# openblas
set(BLA_VENDER OpenBLAS)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message("-- OpenBLAS found.")
    include_directories(/opt/OpenBLAS/include/)
endif(BLAS_FOUND)

# cublas
set(CUBLASLT_LIBRARY cublasLt)
find_library(CUBLASLT_LIBRARY cublasLt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

## curl
#find_package(CURL REQUIRED)
#
#if(CURL_FOUND)
#    message(STATUS "Found CURL version: ${CURL_VERSION_STRING}")
#    message(STATUS "Using CURL include dir(s): ${CURL_INCLUDE_DIRS}")
#    message(STATUS "Using CURL lib(s): ${CURL_LIBRARIES}")
#else()
#    message(FATAL_ERROR "Could not find CURL")
#endif()

find_path(CURL_INCLUDE_DIR curl.h
		HINTS  /data_1/Clion_project/3rdparty/curl/include/curl
		PATH_SUFFIXES include)
MESSAGE(STATUS "Found curl headers at ${CURL_INCLUDE_DIR}")
find_library(CURL_LIBRARIES curl
		HINTS  /data_1/Clion_project/3rdparty/curl
		PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
MESSAGE(STATUS "Found CURL_LIBRARY at ${CURL_LIBRARY}")

include_directories(${CURL_INCLUDE_DIR})

AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
message("DIR_SRCS = ${DIR_SRCS}")
message("CURL_LIBRARIES = ${CURL_LIBRARIES}")
#file(GLOB CU_SRC_FILE "${CMAKE_SOURCE_DIR}/src/*.cu")
#message("CU_SRC_FILE = ${CU_SRC_FILE}")
#cuda_add_executable(${PROJECT_NAME} ${DIR_SRCS} ${CU_SRC_FILE})
cuda_add_executable(${PROJECT_NAME} ${DIR_SRCS})

target_link_libraries(${PROJECT_NAME}
    ${TENSORRT_LIBRARY}
    ${OpenCV_LIBS}
    ${BLAS_LIBRARIES}
    ${CUBLASLT_LIBRARY}
    ${CURL_LIBRARIES}
)
