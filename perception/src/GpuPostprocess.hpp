#pragma once

#include "YoloCandidate.hpp"

#include <cuda_runtime_api.h>

/*
 * GPU 후처리 커널 (--gpu-postprocess, TensorRT 빌드 전용). 구현은 GpuPostprocess.cu
 *
 * 이 헤더는 perception/src 내부 전용임. CUDA 문법이 없어서 g++ 로 컴파일되는 TensorRTBackend.cpp 가 include 함
 * 출력 텐서는 TensorRT 가 쓴 장치 버퍼를 그대로 읽고, 결과는 후보 N 개 자리의 flags·records 에 씀 (D2H 는 TensorRTBackend.cpp 가 함)
 */

// 커널 한 번: 후보(column)마다 scanYoloCandidate() → flags[column] = 1/0, 문턱을 넘은 후보만 records[column] 을 채움
// flags·records 는 장치 메모리, 길이 layout.columns. stream 에 넣기만 하고 기다리지 않음. 실행 설정 오류는 돌려준 값으로 알림
cudaError_t launchScanCandidatesKernel(const float* output, const YoloOutputLayout& layout, float threshold,
                                       unsigned char* flags, YoloCandidate* records, cudaStream_t stream);