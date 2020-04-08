CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)
PROJECT(fb_cuda LANGUAGES CXX CUDA)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE_DIRECTORIES(/usr/local/cuda/include)

ADD_LIBRARY(fb_cuda STATIC
  ./source/filterbank.cpp
  ./include/filterbank.hpp
  ./source/fb_multi_channel_Impl.cu
  ./include/fb_multi_channel_Impl.cuh
  )

TARGET_COMPILE_FEATURES(fb_cuda PUBLIC cxx_std_11)

CUDA_ADD_CUFFT_TO_TARGET(fb_cuda)
 
SET_TARGET_PROPERTIES(fb_cuda
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
ADD_EXECUTABLE(fb ./source/main.cpp)
 
SET_PROPERTY(TARGET fb 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#SET(CMAKE_CUDA_FLAGS "$ {CMAKE_CUDA_FLAGS} -gencode arch=compute_62,code=sm_62")

#SET(CMAKE_BUILD_TYPE Release)

TARGET_LINK_LIBRARIES(fb PRIVATE fb_cuda)
