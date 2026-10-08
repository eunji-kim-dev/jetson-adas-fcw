#pragma once

#include "LetterboxPixel.hpp"
#include "YoloCommon.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <vector>

/*
 * GPU 전처리(--gpu-preprocess)용 계수표·값표와 CPU 기준 함수
 *
 * 이 헤더는 perception/src 내부 전용임. CUDA 없이 컴파일되므로 x86 빌드에도 들어감
 * 계수표는 (원본 크기, resize 크기)가 같으면 매 프레임 같아서 크기가 바뀔 때만 다시 만듦
 */

struct ResizeTables {
    cv::Size source;
    cv::Size resized;
    int mode = kLetterboxCopy;
    std::vector<int> columns;   // 선형: resized.width × 4 (sx, 다음 열, c0, c1). 다른 방식이면 비어 있음
    std::vector<int> rows;      // 선형: resized.height × 4 (sy0, sy1, b0, b1). 다른 방식이면 비어 있음
};

// OpenCV cv::resize(INTER_LINEAR, 8비트) 와 같은 순서·같은 식으로 계수표를 만듦
ResizeTables buildResizeTables(const cv::Size& source, const cv::Size& resized);

// blobFromImage(1/255) 가 0~255 에 주는 float 값 256개. 이 장비의 OpenCV 로 직접 계산함 (GPU 커널의 값표)
std::array<float, 256> buildBlobLut();

// 커널 인자 채우기. source·columns·rows 는 호스트 주소(기준 함수)나 장치 주소(GPU) 둘 다 받음
LetterboxKernelArgs makeLetterboxArgs(const ResizeTables& tables, const LetterboxGeometry& geometry, const cv::Size& inputSize,
                                      const unsigned char* source, std::size_t sourcePitch, const int* columns, const int* rows);

// CPU 기준 함수: GPU 커널과 같은 letterboxPixel() 로 NCHW RGB float 입력 텐서를 만듦 (output 은 3 × H × W 개)
// letterbox() + blobFromImage(1/255, swapRB) 와 비트 단위로 같아야 함. 다르면 그 장비 OpenCV 가 일반 식과 다른 것 (HAL 등)
void letterboxBlobReference(const cv::Mat& bgr, const cv::Size& inputSize, float* output);

// v2 기준 함수: 카메라 YUYV 버퍼(호스트 주소)의 roi 영역으로 GPU 커널(letterboxPixelYuyv)과 같은 식의 입력 텐서를 만듦
// cvtColor(YUV2BGR_YUYV) → letterbox() → blobFromImage 와 비트 단위로 같아야 함. 다르면 그 장비 OpenCV 의 색 변환·resize 가 일반 식과 다른 것
void letterboxBlobReferenceYuyv(const unsigned char* yuyv, std::size_t pitch, const cv::Rect& roi, const cv::Size& inputSize, float* output);