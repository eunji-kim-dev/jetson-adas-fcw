#include "GpuPreprocess.hpp"

/*
 * GPU 전처리 커널 — letterbox(resize·pad) + BGR→RGB + /255 + NCHW 를 한 번에 해서 TensorRT 입력 버퍼에 바로 씀
 *
 * 픽셀 계산은 CPU 기준 함수와 같은 letterboxPixel() (LetterboxPixel.hpp) 이라 OpenCV CPU 경로와 비트 단위로 같아야 함
 * /255 는 곱셈 대신 값표를 씀 — 컴파일 옵션(--use_fast_math 등)에 따라 float 결과가 바뀌는 걸 막으려고
 */

namespace {

// blobFromImage(1/255) 값표. uploadBlobLut() 로 채움
__constant__ float c_blobLut[256];

constexpr int kBlockWidth = 32;
constexpr int kBlockHeight = 8;   // 32 × 8 = 256 스레드 = 값표 칸 수 (블록마다 공유 메모리로 한 번에 옮김)

__global__ void letterboxKernel(LetterboxKernelArgs args, float* output) {
    // 상수 메모리는 같은 워프가 다른 칸을 읽으면 느려져서 블록마다 공유 메모리로 옮겨 씀
    __shared__ float lut[256];
    const int thread = threadIdx.y * blockDim.x + threadIdx.x;
    lut[thread] = c_blobLut[thread];
    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= args.outputWidth || y >= args.outputHeight) return;

    unsigned char blue = 0;
    unsigned char green = 0;
    unsigned char red = 0;
    letterboxPixel(args, x, y, blue, green, red);

    const size_t plane = static_cast<size_t>(args.outputWidth) * static_cast<size_t>(args.outputHeight);
    const size_t index = static_cast<size_t>(y) * static_cast<size_t>(args.outputWidth) + static_cast<size_t>(x);
    output[index] = lut[red];
    output[plane + index] = lut[green];
    output[2 * plane + index] = lut[blue];
}

} // namespace

cudaError_t uploadBlobLut(const float* lut256) {
    return cudaMemcpyToSymbol(c_blobLut, lut256, sizeof(float) * 256);
}

cudaError_t launchLetterboxKernel(const LetterboxKernelArgs& args, float* output, cudaStream_t stream) {
    const dim3 block(kBlockWidth, kBlockHeight);
    const dim3 grid((args.outputWidth + kBlockWidth - 1) / kBlockWidth, (args.outputHeight + kBlockHeight - 1) / kBlockHeight);
    letterboxKernel<<<grid, block, 0, stream>>>(args, output);
    return cudaGetLastError();
}

namespace {

// v2: 원본이 카메라 YUYV 버퍼. letterboxKernel 과 같고 원본 픽셀을 읽을 때만 색 변환함
__global__ void letterboxYuyvKernel(LetterboxKernelArgs args, float* output) {
    __shared__ float lut[256];
    const int thread = threadIdx.y * blockDim.x + threadIdx.x;
    lut[thread] = c_blobLut[thread];
    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= args.outputWidth || y >= args.outputHeight) return;

    unsigned char blue = 0;
    unsigned char green = 0;
    unsigned char red = 0;
    letterboxPixelYuyv(args, x, y, blue, green, red);

    const size_t plane = static_cast<size_t>(args.outputWidth) * static_cast<size_t>(args.outputHeight);
    const size_t index = static_cast<size_t>(y) * static_cast<size_t>(args.outputWidth) + static_cast<size_t>(x);
    output[index] = lut[red];
    output[plane + index] = lut[green];
    output[2 * plane + index] = lut[blue];
}

} // namespace

cudaError_t launchLetterboxYuyvKernel(const LetterboxKernelArgs& args, float* output, cudaStream_t stream) {
    const dim3 block(kBlockWidth, kBlockHeight);
    const dim3 grid((args.outputWidth + kBlockWidth - 1) / kBlockWidth, (args.outputHeight + kBlockHeight - 1) / kBlockHeight);
    letterboxYuyvKernel<<<grid, block, 0, stream>>>(args, output);
    return cudaGetLastError();
}