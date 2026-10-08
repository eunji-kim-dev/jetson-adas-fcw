#pragma once

#include "LetterboxPixel.hpp"

#include <cuda_runtime_api.h>

/*
 * GPU 전처리 커널 (--gpu-preprocess, TensorRT 빌드 전용). 구현은 GpuPreprocess.cu
 *
 * 이 헤더는 perception/src 내부 전용임. CUDA 문법이 없어서 g++ 로 컴파일되는 TensorRTBackend.cpp 가 include 함
 * 프레임 업로드·계수표 관리는 TensorRTBackend.cpp 가 하고, 여기는 커널 실행과 값표 올리기만 함
 */

// blobFromImage(1/255) 값표 256개를 GPU 상수 메모리에 올림 (buildBlobLut() 결과). 커널 실행 전에 한 번
cudaError_t uploadBlobLut(const float* lut256);

// 커널 한 번: 출력 (1, 3, H, W) 의 픽셀마다 letterboxPixel() → 값표 → R·G·B 평면에 씀 (blobFromImage swapRB=true 순서)
// stream 에 넣기만 하고 기다리지 않음. 실행 설정 오류는 돌려준 값으로 알림
cudaError_t launchLetterboxKernel(const LetterboxKernelArgs& args, float* output, cudaStream_t stream);

// v2 (--camera-zero-copy): 원본이 카메라 YUYV 버퍼(CUDA 등록한 MMAP 버퍼의 GPU 주소)인 커널. 픽셀마다 letterboxPixelYuyv()
// 값표·출력 순서는 위 커널과 같음. stream 에 넣기만 하고 기다리지 않음
cudaError_t launchLetterboxYuyvKernel(const LetterboxKernelArgs& args, float* output, cudaStream_t stream);