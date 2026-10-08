#include "LetterboxTables.hpp"

#include <opencv2/dnn.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

// OpenCV resize.cpp 의 INTER_RESIZE_COEF_SCALE (1 << 11). 공개 헤더에 없어서 같은 값을 둠
constexpr int resizeCoefficientScale = 2048;

} // namespace

/*
 * cv::resize(INTER_LINEAR, 8비트) 의 계수표를 같은 식으로 만듦 (OpenCV 4.x imgproc/src/resize.cpp)
 *   scale = 1 / (resize 크기 ÷ 원본 크기)  (double)
 *   좌표  = (float)((d + 0.5) · scale − 0.5), s = floor, f = 좌표 − s
 *   가로 경계: s < 0 이면 s = 0, f = 0 / s ≥ W − 1 이면 s = W − 1, f = 0
 *   세로 경계: f 는 그대로 두고 읽는 두 행만 [0, H − 1] 로 자름
 *   계수  = saturate_cast<short>((1 − f) · 2048), saturate_cast<short>(f · 2048)  (짝수 쪽 반올림)
 *   가로·세로 정확히 2배 축소면 OpenCV 가 INTER_AREA 빠른 경로로 바꿈 → kLetterboxArea2x
 */
ResizeTables buildResizeTables(const cv::Size& source, const cv::Size& resized) {
    if (source.width <= 0 || source.height <= 0 || resized.width <= 0 || resized.height <= 0) {
        throw std::invalid_argument("resize 계수표: 크기가 0 임");
    }

    ResizeTables tables;
    tables.source = source;
    tables.resized = resized;

    // 같은 크기면 cv::resize 가 복사만 함
    if (resized == source) {
        tables.mode = kLetterboxCopy;
        return tables;
    }

    const double inverseScaleX = static_cast<double>(resized.width) / source.width;
    const double inverseScaleY = static_cast<double>(resized.height) / source.height;
    const double scaleX = 1.0 / inverseScaleX;
    const double scaleY = 1.0 / inverseScaleY;

    // 가로·세로 축척이 정확히 2 면 INTER_AREA 빠른 경로 (OpenCV 주석: 이때 INTER_AREA 와 INTER_LINEAR 가 같음)
    const int integerScaleX = cv::saturate_cast<int>(scaleX);
    const int integerScaleY = cv::saturate_cast<int>(scaleY);
    const bool areaFast = std::abs(scaleX - integerScaleX) < DBL_EPSILON && std::abs(scaleY - integerScaleY) < DBL_EPSILON;
    if (areaFast && integerScaleX == 2 && integerScaleY == 2) {
        tables.mode = kLetterboxArea2x;
        return tables;
    }

    tables.mode = kLetterboxLinear;
    tables.columns.resize(static_cast<std::size_t>(resized.width) * 4);
    for (int dx = 0; dx < resized.width; ++dx) {
        float fx = static_cast<float>((dx + 0.5) * scaleX - 0.5);
        int sx = cvFloor(fx);
        fx -= static_cast<float>(sx);
        if (sx < 0) {
            fx = 0.0F;
            sx = 0;
        }
        if (sx >= source.width - 1) {
            fx = 0.0F;
            sx = source.width - 1;
        }
        int* column = tables.columns.data() + 4 * dx;
        column[0] = sx;
        column[1] = std::min(sx + 1, source.width - 1);   // sx = W − 1 이면 c1 = 0 이라 값에 안 들어감 (읽기만 안전하게)
        column[2] = cv::saturate_cast<short>((1.0F - fx) * resizeCoefficientScale);
        column[3] = cv::saturate_cast<short>(fx * resizeCoefficientScale);
    }

    tables.rows.resize(static_cast<std::size_t>(resized.height) * 4);
    for (int dy = 0; dy < resized.height; ++dy) {
        float fy = static_cast<float>((dy + 0.5) * scaleY - 0.5);
        const int sy = cvFloor(fy);
        fy -= static_cast<float>(sy);
        int* row = tables.rows.data() + 4 * dy;
        row[0] = std::clamp(sy, 0, source.height - 1);
        row[1] = std::clamp(sy + 1, 0, source.height - 1);
        row[2] = cv::saturate_cast<short>((1.0F - fy) * resizeCoefficientScale);
        row[3] = cv::saturate_cast<short>(fy * resizeCoefficientScale);
    }
    return tables;
}

// blobFromImage(1/255) 를 0~255 한 줄에 직접 돌려서 값을 받음
// 식((float)v / 255 또는 v × (float)(1/255))을 따로 쓰지 않고 이 장비의 OpenCV 가 내는 값을 그대로 씀
std::array<float, 256> buildBlobLut() {
    cv::Mat values(1, 256, CV_8UC1);
    for (int v = 0; v < 256; ++v) values.at<unsigned char>(0, v) = static_cast<unsigned char>(v);
    const cv::Mat blob = cv::dnn::blobFromImage(values, 1.0 / 255.0, cv::Size(256, 1), cv::Scalar(), false, false);
    if (!blob.isContinuous() || blob.total() != 256) throw std::runtime_error("blob 값표 계산 실패");
    std::array<float, 256> lut{};
    std::memcpy(lut.data(), blob.ptr<float>(), sizeof(float) * lut.size());
    return lut;
}

LetterboxKernelArgs makeLetterboxArgs(const ResizeTables& tables, const LetterboxGeometry& geometry, const cv::Size& inputSize,
                                      const unsigned char* source, std::size_t sourcePitch, const int* columns, const int* rows) {
    LetterboxKernelArgs args;
    args.source = source;
    args.sourcePitch = sourcePitch;
    args.sourceWidth = tables.source.width;
    args.sourceHeight = tables.source.height;
    args.mode = tables.mode;
    args.columnTable = columns;
    args.rowTable = rows;
    args.resizedWidth = geometry.resized.width;
    args.resizedHeight = geometry.resized.height;
    args.padLeft = geometry.padLeft;
    args.padTop = geometry.padTop;
    args.outputWidth = inputSize.width;
    args.outputHeight = inputSize.height;
    return args;
}

void letterboxBlobReference(const cv::Mat& bgr, const cv::Size& inputSize, float* output) {
    if (bgr.type() != CV_8UC3) throw std::invalid_argument("기준 함수는 8비트 BGR 3채널만 받음");
    static const std::array<float, 256> lut = buildBlobLut();

    const LetterboxGeometry geometry = computeLetterboxGeometry(bgr.size(), inputSize);
    const ResizeTables tables = buildResizeTables(bgr.size(), geometry.resized);
    const LetterboxKernelArgs args = makeLetterboxArgs(tables, geometry, inputSize, bgr.data, bgr.step,
                                                       tables.columns.data(), tables.rows.data());

    // blobFromImage(swapRB=true) 와 같은 순서: R 평면, G 평면, B 평면
    const std::size_t plane = static_cast<std::size_t>(inputSize.width) * static_cast<std::size_t>(inputSize.height);
    for (int y = 0; y < inputSize.height; ++y) {
        for (int x = 0; x < inputSize.width; ++x) {
            unsigned char blue = 0, green = 0, red = 0;
            letterboxPixel(args, x, y, blue, green, red);
            const std::size_t index = static_cast<std::size_t>(y) * inputSize.width + x;
            output[index] = lut[red];
            output[plane + index] = lut[green];
            output[2 * plane + index] = lut[blue];
        }
    }
}

void letterboxBlobReferenceYuyv(const unsigned char* yuyv, std::size_t pitch, const cv::Rect& roi, const cv::Size& inputSize, float* output) {
    if (yuyv == nullptr || roi.width <= 0 || roi.height <= 0) throw std::invalid_argument("YUYV 기준 함수: 버퍼가 없거나 roi 가 비었음");
    static const std::array<float, 256> lut = buildBlobLut();

    const LetterboxGeometry geometry = computeLetterboxGeometry(roi.size(), inputSize);
    const ResizeTables tables = buildResizeTables(roi.size(), geometry.resized);
    LetterboxKernelArgs args = makeLetterboxArgs(tables, geometry, inputSize, yuyv + static_cast<std::size_t>(roi.y) * pitch, pitch,
                                                 tables.columns.data(), tables.rows.data());
    args.sourceOffsetX = roi.x;

    // blobFromImage(swapRB=true) 와 같은 순서: R 평면, G 평면, B 평면
    const std::size_t plane = static_cast<std::size_t>(inputSize.width) * static_cast<std::size_t>(inputSize.height);
    for (int y = 0; y < inputSize.height; ++y) {
        for (int x = 0; x < inputSize.width; ++x) {
            unsigned char blue = 0, green = 0, red = 0;
            letterboxPixelYuyv(args, x, y, blue, green, red);
            const std::size_t index = static_cast<std::size_t>(y) * inputSize.width + x;
            output[index] = lut[red];
            output[plane + index] = lut[green];
            output[2 * plane + index] = lut[blue];
        }
    }
}