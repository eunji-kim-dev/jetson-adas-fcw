#pragma once

#include <cstddef>

/*
 * letterbox 한 픽셀 계산 — CPU 기준 함수(LetterboxTables.cpp)와 GPU 커널(GpuPreprocess.cu)이 같은 함수를 씀
 *
 * OpenCV 4.x 의 8비트 cv::resize(INTER_LINEAR) + copyMakeBorder(114) 를 정수식 그대로 재현함 (imgproc/src/resize.cpp)
 *   선형    : 가로 h = S[sx]·c0 + S[sx+1]·c1, 세로 (((b0·(h0>>4))>>16) + ((b1·(h1>>4))>>16) + 2) >> 2
 *   2배 축소: 가로·세로 정확히 2배면 OpenCV 가 INTER_AREA 빠른 경로로 바꿈 → (a + b + c + d + 2) >> 2
 *   같은 크기: 복사
 * 계수표(sx·c0·c1, sy·b0·b1)는 LetterboxTables.cpp 가 OpenCV 와 같은 식으로 미리 만듦
 * 이 파일은 OpenCV 없이 g++·nvcc 둘 다에서 컴파일됨
 */

#ifdef __CUDACC__
#define LETTERBOX_HOST_DEVICE __host__ __device__
#else
#define LETTERBOX_HOST_DEVICE
#endif

// resize 방식. 값은 커널 인자로 그대로 넘김
enum LetterboxResizeMode : int {
    kLetterboxCopy = 0,     // resize 크기 == 원본 크기 → 복사
    kLetterboxArea2x = 1,   // 가로·세로 정확히 2배 축소 → 2×2 평균
    kLetterboxLinear = 2,   // 그 밖 → 정수 선형 보간
};

// 여백 회색값 (copyMakeBorder 의 Scalar(114, 114, 114))
constexpr unsigned char kLetterboxPadValue = 114;

struct LetterboxKernelArgs {
    const unsigned char* source = nullptr;   // BGR 8비트 원본. 행 간격 sourcePitch 바이트 (ROI 뷰면 원래 영상의 step)
    std::size_t sourcePitch = 0;
    int sourceWidth = 0;
    int sourceHeight = 0;
    int mode = kLetterboxCopy;
    const int* columnTable = nullptr;        // 선형: resize 열마다 4칸 (sx, 다음 열, c0, c1)
    const int* rowTable = nullptr;           // 선형: resize 행마다 4칸 (sy0, sy1, b0, b1). 두 행은 [0, H-1] 로 자른 값
    int resizedWidth = 0;
    int resizedHeight = 0;
    int padLeft = 0;
    int padTop = 0;
    int outputWidth = 0;                     // 모델 입력 가로 (예: 640)
    int outputHeight = 0;                    // 모델 입력 세로 (예: 640, 288)
    int sourceOffsetX = 0;                   // YUYV(v2) 전용: source 0열이 카메라 영상의 몇 번째 열인지 (Crop 창 x). U·V 는 절대 열 짝(0-1, 2-3 …)으로 묶임
};

// 출력(모델 입력) 좌표 (x, y) 의 BGR 8비트 값. 여백이면 114
LETTERBOX_HOST_DEVICE inline void letterboxPixel(const LetterboxKernelArgs& args, int x, int y,
                                                unsigned char& blue, unsigned char& green, unsigned char& red) {
    const int rx = x - args.padLeft;
    const int ry = y - args.padTop;
    if (rx < 0 || ry < 0 || rx >= args.resizedWidth || ry >= args.resizedHeight) {
        blue = kLetterboxPadValue;
        green = kLetterboxPadValue;
        red = kLetterboxPadValue;
        return;
    }

    unsigned char value[3];
    if (args.mode == kLetterboxCopy) {
        const unsigned char* pixel = args.source + static_cast<std::size_t>(ry) * args.sourcePitch + static_cast<std::size_t>(rx) * 3;
        value[0] = pixel[0];
        value[1] = pixel[1];
        value[2] = pixel[2];
    } else if (args.mode == kLetterboxArea2x) {
        // OpenCV ResizeAreaFastVec (cn=3): (위 왼쪽 + 위 오른쪽 + 아래 왼쪽 + 아래 오른쪽 + 2) >> 2
        const unsigned char* top = args.source + static_cast<std::size_t>(2 * ry) * args.sourcePitch + static_cast<std::size_t>(2 * rx) * 3;
        const unsigned char* bottom = top + args.sourcePitch;
        for (int channel = 0; channel < 3; ++channel) {
            value[channel] = static_cast<unsigned char>((top[channel] + top[channel + 3] + bottom[channel] + bottom[channel + 3] + 2) >> 2);
        }
    } else {
        // OpenCV HResizeLinear(8u → int) + VResizeLinear(int → 8u, FixedPtCast) 와 같은 정수식
        const int* column = args.columnTable + 4 * rx;
        const int* row = args.rowTable + 4 * ry;
        const unsigned char* row0 = args.source + static_cast<std::size_t>(row[0]) * args.sourcePitch;
        const unsigned char* row1 = args.source + static_cast<std::size_t>(row[1]) * args.sourcePitch;
        const int x0 = column[0] * 3;
        const int x1 = column[1] * 3;
        const int c0 = column[2];
        const int c1 = column[3];
        const int b0 = row[2];
        const int b1 = row[3];
        for (int channel = 0; channel < 3; ++channel) {
            const int h0 = row0[x0 + channel] * c0 + row0[x1 + channel] * c1;
            const int h1 = row1[x0 + channel] * c0 + row1[x1 + channel] * c1;
            value[channel] = static_cast<unsigned char>((((b0 * (h0 >> 4)) >> 16) + ((b1 * (h1 >> 4)) >> 16) + 2) >> 2);
        }
    }
    blue = value[0];
    green = value[1];
    red = value[2];
}

// ------------------------------------------------------------
// GPU 전처리 v2 (--camera-zero-copy): 카메라 YUYV 버퍼를 직접 읽음
// ------------------------------------------------------------

// YUYV 4:2:2 한 픽셀 → BGR. OpenCV cvtColor(COLOR_YUV2BGR_YUYV) 와 같은 정수식 (imgproc/src/color_yuv.simd.hpp, BT.601, 20비트 고정소수점)
//   y = max(Y − 16, 0) · 1220542
//   B = (y + 2^19 + 2116026·(U − 128)) >> 20
//   G = (y + 2^19 − 852492·(V − 128) − 409993·(U − 128)) >> 20
//   R = (y + 2^19 + 1673527·(V − 128)) >> 20   → 셋 다 0~255 로 자름
// row 는 그 줄의 시작, x 는 카메라 영상의 절대 열. 두 픽셀이 U·V 를 같이 씀: [Y0 U Y1 V]
LETTERBOX_HOST_DEVICE inline void yuyvToBgr(const unsigned char* row, int x, unsigned char* bgr) {
    const unsigned char* pair = row + static_cast<std::size_t>(x >> 1) * 4;
    const int luma = pair[(x & 1) * 2];
    const int u = static_cast<int>(pair[1]) - 128;
    const int v = static_cast<int>(pair[3]) - 128;
    const int y = (luma > 16 ? luma - 16 : 0) * 1220542;
    const int half = 1 << 19;
    const int b = (y + half + 2116026 * u) >> 20;
    const int g = (y + half - 852492 * v - 409993 * u) >> 20;
    const int r = (y + half + 1673527 * v) >> 20;
    bgr[0] = static_cast<unsigned char>(b < 0 ? 0 : (b > 255 ? 255 : b));
    bgr[1] = static_cast<unsigned char>(g < 0 ? 0 : (g > 255 ? 255 : g));
    bgr[2] = static_cast<unsigned char>(r < 0 ? 0 : (r > 255 ? 255 : r));
}

// letterboxPixel() 과 같은 계산인데 원본이 YUYV 임. 원본 픽셀을 읽는 자리에서 yuyvToBgr() 로 바꿔 읽음
// 색 변환은 픽셀마다 따로라 "cvtColor 로 BGR 을 다 만든 뒤 letterbox" 와 비트 단위로 같음
// args.source 는 roi 첫 줄 시작 (YUYV 버퍼 + roi.y · pitch), args.sourceOffsetX = roi.x
LETTERBOX_HOST_DEVICE inline void letterboxPixelYuyv(const LetterboxKernelArgs& args, int x, int y,
                                                    unsigned char& blue, unsigned char& green, unsigned char& red) {
    const int rx = x - args.padLeft;
    const int ry = y - args.padTop;
    if (rx < 0 || ry < 0 || rx >= args.resizedWidth || ry >= args.resizedHeight) {
        blue = kLetterboxPadValue;
        green = kLetterboxPadValue;
        red = kLetterboxPadValue;
        return;
    }

    unsigned char value[3];
    if (args.mode == kLetterboxCopy) {
        yuyvToBgr(args.source + static_cast<std::size_t>(ry) * args.sourcePitch, args.sourceOffsetX + rx, value);
    } else if (args.mode == kLetterboxArea2x) {
        // (위 왼쪽 + 위 오른쪽 + 아래 왼쪽 + 아래 오른쪽 + 2) >> 2
        const unsigned char* top = args.source + static_cast<std::size_t>(2 * ry) * args.sourcePitch;
        const unsigned char* bottom = top + args.sourcePitch;
        const int sx = args.sourceOffsetX + 2 * rx;
        unsigned char topLeft[3], topRight[3], bottomLeft[3], bottomRight[3];
        yuyvToBgr(top, sx, topLeft);
        yuyvToBgr(top, sx + 1, topRight);
        yuyvToBgr(bottom, sx, bottomLeft);
        yuyvToBgr(bottom, sx + 1, bottomRight);
        for (int channel = 0; channel < 3; ++channel) {
            value[channel] = static_cast<unsigned char>((topLeft[channel] + topRight[channel] + bottomLeft[channel] + bottomRight[channel] + 2) >> 2);
        }
    } else {
        // 선형: 두 줄 × 두 열을 읽어 letterboxPixel() 과 같은 정수식
        const int* column = args.columnTable + 4 * rx;
        const int* row = args.rowTable + 4 * ry;
        const unsigned char* row0 = args.source + static_cast<std::size_t>(row[0]) * args.sourcePitch;
        const unsigned char* row1 = args.source + static_cast<std::size_t>(row[1]) * args.sourcePitch;
        const int x0 = args.sourceOffsetX + column[0];
        const int x1 = args.sourceOffsetX + column[1];
        unsigned char p00[3], p01[3], p10[3], p11[3];
        yuyvToBgr(row0, x0, p00);
        yuyvToBgr(row0, x1, p01);
        yuyvToBgr(row1, x0, p10);
        yuyvToBgr(row1, x1, p11);
        const int c0 = column[2];
        const int c1 = column[3];
        const int b0 = row[2];
        const int b1 = row[3];
        for (int channel = 0; channel < 3; ++channel) {
            const int h0 = p00[channel] * c0 + p01[channel] * c1;
            const int h1 = p10[channel] * c0 + p11[channel] * c1;
            value[channel] = static_cast<unsigned char>((((b0 * (h0 >> 4)) >> 16) + ((b1 * (h1 >> 4)) >> 16) + 2) >> 2);
        }
    }
    blue = value[0];
    green = value[1];
    red = value[2];
}