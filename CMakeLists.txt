CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)
PROJECT(fb_cuda LANGUAGES CXX CUDA)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

FIND_PACKAGE(CUDA REQUIRED)

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
TARGET_LINK_LIBRARIES(fb PRIVATE fb_cuda)