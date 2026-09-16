#include "perception/InferenceBackend.hpp"

#include "OpenCVDNNBackend.hpp"
#ifdef PERCEPTION_HAS_TENSORRT
#include "TensorRTBackend.hpp"
#endif

#include <memory>
#include <stdexcept>
#include <string>

// 백엔드 이름 → 구현체 매핑은 이 파일에서만 관리함
// 새 백엔드는 여기에 분기 하나를 추가하면 되고, 앱과 YoloDetector 는 수정하지 않음
//
//   opencv_dnn      OpenCV DNN, CPU
//   tensorrt_fp32   TensorRT, GPU, 32비트  (ENABLE_TENSORRT=ON 빌드에서만)
//   tensorrt_fp16   TensorRT, GPU, 16비트  (ENABLE_TENSORRT=ON 빌드에서만)
std::unique_ptr<InferenceBackend> createInferenceBackend(
    const std::string& backendName,
    const std::string& modelPath,
    float confidenceThreshold,
    float nmsThreshold
) {
    if (backendName == "opencv_dnn") {
        return std::make_unique<OpenCVDNNBackend>(modelPath, confidenceThreshold, nmsThreshold);
    }

    // "tensorrt_<정밀도>" 를 정밀도 문자열로 나눔
    const std::string prefix = "tensorrt_";
    if (backendName.rfind(prefix, 0) == 0) {
        const std::string precision = backendName.substr(prefix.size());
#ifdef PERCEPTION_HAS_TENSORRT
        return std::make_unique<TensorRTBackend>(modelPath, precision, confidenceThreshold, nmsThreshold);
#else
        throw std::invalid_argument("이 빌드에는 TensorRT 가 없음 (-DENABLE_TENSORRT=ON 으로 다시 빌드): " + backendName);
#endif
    }

    throw std::invalid_argument("지원하지 않는 backend: " + backendName + " (사용 가능: opencv_dnn, tensorrt_fp32, tensorrt_fp16)");
}