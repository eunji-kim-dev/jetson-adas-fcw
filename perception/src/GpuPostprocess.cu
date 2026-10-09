#include "GpuPostprocess.hpp"

/*
 * GPU 후처리 커널 — YOLO 출력 텐서 (1, C, N) 을 후보마다 훑어 클래스 최댓값을 찾고 문턱과 비교함
 *
 * 후보 계산은 CPU 기준 함수와 같은 scanYoloCandidate() (YoloCandidate.hpp) 이고 비교만 하므로 결과가 비트 단위로 같음
 * 스레드 하나 = 후보 하나. (1, C, N) 배치라 같은 채널의 이웃 후보가 이웃 주소 → 워프가 한 줄로 읽음 (coalesced)
 * CPU 는 flags 가 1 인 records 만 후보 번호 순으로 모아 좌표 복원·NMS 를 함 → 출력 텐서 전체(Full 2.8 MB) 대신 N × 29 바이트만 D2H 함
 */

namespace {

constexpr int kBlockSize = 256;

__global__ void scanCandidatesKernel(const float* output, YoloOutputLayout layout, float threshold, unsigned char* flags, YoloCandidate* records) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= layout.columns) return;

    YoloCandidate candidate;
    const bool kept = scanYoloCandidate(output, layout, column, threshold, candidate);
    flags[column] = kept ? 1 : 0;
    // 문턱을 못 넘은 자리는 안 씀 (지난 프레임 값이 남지만 flags 가 0 이라 CPU 가 읽지 않음)
    if (kept) records[column] = candidate;
}

} // namespace

cudaError_t launchScanCandidatesKernel(const float* output, const YoloOutputLayout& layout, float threshold,
                                       unsigned char* flags, YoloCandidate* records, cudaStream_t stream) {
    const dim3 block(kBlockSize);
    const dim3 grid((layout.columns + kBlockSize - 1) / kBlockSize);
    scanCandidatesKernel<<<grid, block, 0, stream>>>(output, layout, threshold, flags, records);
    return cudaGetLastError();
}