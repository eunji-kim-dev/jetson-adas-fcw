#include "perception/InferenceBackend.hpp"

#include "OpenCVDNNBackend.hpp"

#include <memory>
#include <stdexcept>
#include <string>

// 백엔드 이름 → 구현체 매핑은 이 파일에서만 관리한다.
// 새 백엔드(예: "tensorrt")는 여기에 분기 하나를 추가하면 되고,
// 앱과 YoloDetector는 수정하지 않는다.
std::unique_ptr<InferenceBackend> createInferenceBackend(
    const std::string& backendName,
    const std::string& modelPath,
    float confidenceThreshold,
    float nmsThreshold
) {
    if (backendName == "opencv_dnn") {
        return std::make_unique<OpenCVDNNBackend>(modelPath, confidenceThreshold, nmsThreshold);
    }
    throw std::invalid_argument("지원하지 않는 backend: " + backendName + " (사용 가능: opencv_dnn)");
}
