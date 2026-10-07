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