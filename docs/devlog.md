# Development Log

Blackbox ADAS 프로젝트를 구현하면서 진행한 기능 개발과 그 과정에서의 기술적 판단을 기록합니다.

문서에 적힌 임계값은 모두 `src/` 안의 실제 상수와 대조해 맞춘 값입니다. 계측한 수치와 눈으로만 확인한 관찰은 구분해서 표기했습니다.

## 1. 프로젝트 목표

블랙박스 영상을 입력받아 차량과 보행자를 검출하고, 프레임 간 객체 이동을 추적해 주행 경로 안의 위험 상황을 판단하는 C++ 영상처리 파이프라인을 만드는 것이 목표였습니다.

초기에 잡은 전체 흐름은 다음과 같았습니다.

```
블랙박스 영상 입력
→ 프레임 전처리
→ YOLO 객체 검출
→ 객체 추적 및 ID 부여
→ 주행 ROI 진입 판정
→ 위험 이벤트 판단
→ 결과 영상 및 성능 정보 출력
```

작업을 진행하면서 ROI 진입 여부만 보는 것으로는 부족하다고 판단했고, 자차 차선의 선행 차량 한 대를 선택해 바운딩 박스 크기 변화로 TTC-P를 계산하는 구조까지 발전시켰습니다. 최종적으로 정리된 흐름은 다음과 같습니다.

```
블랙박스 영상 입력
→ Letterbox 전처리
→ YOLOv8 ONNX 전체 프레임 추론
→ 원거리 도로 영역 crop 보조 추론
→ 두 결과 병합
→ 차량 그룹 단위 NMS
→ 병합 후 2차 NMS
→ 포함 관계 중복 제거
→ 극소 박스 제거
→ Kalman Filter 기반 다중 객체 추적
→ 3단계 Detection 매칭
→ 도로 ROI와 ego lane 분리
→ 선행 차량 LEAD 선택 (체류 조건 + 추월 차량 제외)
→ 바운딩 박스 높이 변화 이력 수집
→ TTC-P 계산
→ 위험 단계 안정화 및 오경고 필터링
→ 장면 전환 감지 및 상태 초기화
→ 한국어 상태 정보와 경고 출력
→ 결과 영상 저장 및 성능 정보 출력
```

**사용 기술**

* C++17
* OpenCV
* OpenCV DNN
* OpenCV FreeType
* YOLOv8 ONNX
* Kalman Filter
* IoU·중심 거리·박스 크기 변화를 결합한 Cost-based Greedy Matching
* CMake
* Ubuntu Linux
* Git / GitHub

---

## 2. 영상 입출력 파이프라인

가장 먼저 `VideoCapture`와 `VideoWriter`로 프레임 단위 입출력 구조부터 잡았습니다.

```
입력 영상 경로 확인
→ 영상 열기
→ 해상도와 FPS 확인
→ 프레임 반복 처리
→ 결과 영상 저장
→ 처리 프레임 수와 평균 속도 출력
```

처리 시간은 `std::chrono::steady_clock`으로 측정했습니다.

```cpp
const auto processingStart =
    std::chrono::steady_clock::now();

// 프레임 처리

const auto processingEnd =
    std::chrono::steady_clock::now();
```

원본 영상의 재생 FPS와 프로그램이 실제로 처리하는 FPS는 서로 다른 값이라 따로 관리했습니다. 원본 FPS는 결과 영상 저장과 시간 계산에 쓰고, 처리 FPS는 현재 CPU 환경에서 파이프라인이 얼마나 빠르게 도는지 확인하는 용도로 뒀습니다.

FPS는 시간 관련 임계값 환산에도 쓰입니다. LEAD 체류 시간, 장면 전환 후 분석 유예, 경고 배너 유지 시간이 모두 `sourceFps`를 곱해 프레임 수로 변환됩니다.

```cpp
const int leadEligibilityFrames =
    std::max(1, static_cast<int>(std::round(sourceFps * 0.5)));

const int sceneWarmupFrames =
    std::max(1, static_cast<int>(std::round(sourceFps * 0.5)));

const int warningHoldFrames =
    std::max(1, static_cast<int>(std::round(sourceFps * 0.25)));
```

출력 파일이 생성되지 않거나 짧게 잘려 저장되는 것을 막기 위해, 프로그램 종료 전에 `capture.release()`와 `writer.release()`를 명시적으로 호출하도록 했습니다.

## 3. 주행 경로 ROI

화면에 보이는 모든 객체를 위험 객체로 취급하면 의미가 없기 때문에, 차량이 실제로 지나갈 가능성이 높은 영역을 사다리꼴 ROI로 정의했습니다.

좌표는 고정 픽셀 대신 영상 너비와 높이의 비율로 계산했습니다. 현재 코드에는 §20에서 분리한 두 개의 ROI가 들어 있습니다.

```cpp
// 넓은 도로 관심 영역
const std::vector<cv::Point> roadRoi = {
    cv::Point(width * 0.37F, height * 0.64F),
    cv::Point(width * 0.64F, height * 0.64F),
    cv::Point(width * 0.96F, height * 0.96F),
    cv::Point(width * 0.04F, height * 0.96F)
};

// 실제 위험 판단에 쓰는 자차 차선 영역
const std::vector<cv::Point> egoLaneRoi = {
    cv::Point(width * 0.40F, height * 0.65F),
    cv::Point(width * 0.48F, height * 0.65F),
    cv::Point(width * 0.68F, height * 0.95F),
    cv::Point(width * 0.10F, height * 0.95F)
};
```

비율로 계산했기 때문에 입력 해상도가 달라져도 같은 형태의 사다리꼴이 만들어집니다. 다만 **형태 자체는 현재 테스트 영상의 카메라 장착 위치에 맞춰 손으로 조정한 값입니다.** 차선 검출을 수행하지 않으므로 곡선로, 차선 변경, 카메라 위치가 다른 영상에서는 실제 주행 차선과 어긋납니다. 1280×720 기준으로 `egoLaneRoi` 하단 중심은 x=499이고 화면 중심은 x=640이라, 약 141px 왼쪽으로 치우쳐 있습니다. 이 값은 이 영상에서만 맞습니다. 한계 항목에 별도로 정리했습니다.

시각화는 원본 프레임을 완전히 가리지 않도록 반투명 오버레이로 처리했습니다.

```
ROI 오버레이
+ 원본 프레임
→ addWeighted()로 합성
```

처음에는 ROI 윗변이 지평선 가까이까지 올라가 있어서 너무 많은 원거리 차량이 위험 영역에 포함됐습니다. 영상의 카메라 시점에 맞춰 윗변을 아래로 내려 조정했습니다.

## 4. YOLOv8 ONNX 연동

Python 추론 스크립트에 의존하지 않고 C++ 실행 파일 안에서 검출까지 끝내기 위해, YOLOv8n을 ONNX로 변환해 OpenCV DNN에 붙였습니다.

```cpp
cv::dnn::Net net =
    cv::dnn::readNetFromONNX(
        "models/yolov8n.onnx"
    );
```

실행 환경은 우선 CPU로 고정했습니다.

```cpp
net.setPreferableBackend(
    cv::dnn::DNN_BACKEND_OPENCV
);

net.setPreferableTarget(
    cv::dnn::DNN_TARGET_CPU
);
```

COCO 80개 클래스 중에서는 ADAS 분석과 관련 있는 것만 골라 사용했습니다.

- person (0)
- bicycle (1)
- car (2)
- motorcycle (3)
- bus (5)
- truck (7)

여기에 §27에서 다루는 오분류 복구 때문에 train(6)이 조건부로 한 번 더 등장합니다.

객체 검출 결과는 다음 구조로 정리했습니다.

```cpp
struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
};
```

검출 단계와 추적 단계를 분리하기 위해, YOLO 출력은 전부 Detection 목록으로 변환한 뒤 `MultiObjectTracker`로 넘기는 구조를 잡았습니다.

## 5. Letterbox 전처리와 좌표 복원

블랙박스 영상은 보통 가로가 긴 비율인데 YOLO 입력은 640×640 정사각형입니다. 원본을 그대로 resize하면 차량의 가로세로 비율이 왜곡되기 때문에, 비율은 유지한 채 크기만 줄이고 남는 공간에 패딩을 채우는 Letterbox 방식을 썼습니다.

```
원본 영상 크기 확인
→ 가로세로 비율을 유지한 채 축소
→ 남는 영역에 114 색상 패딩
→ 640×640 입력 생성
```

전처리 결과에는 좌표를 되돌리는 데 필요한 정보도 함께 저장해뒀습니다.

```cpp
struct LetterboxResult {
    cv::Mat image;
    float scale;
    int padX;
    int padY;
};
```

YOLO 출력 좌표는 다음 순서로 원본 영상 좌표로 복원합니다.

```
YOLO 출력 좌표
→ Letterbox 패딩 제거
→ resize 배율로 나누기
→ 원본 영상 좌표로 복원
→ 프레임 범위 안으로 clamp
```

```cpp
left   = std::clamp(left,   0, frame.cols - 1);
top    = std::clamp(top,    0, frame.rows - 1);
right  = std::clamp(right,  0, frame.cols - 1);
bottom = std::clamp(bottom, 0, frame.rows - 1);
```

복원한 뒤 너비나 높이가 0 이하가 된 박스는 버렸습니다.

이 clamp에는 §23의 TTC-P 계산과 충돌하는 부작용이 있습니다. 선행 차량이 매우 가까워져 박스 하단이 화면 밖으로 나가면 `bottom`이 `frame.rows - 1`로 고정되면서 박스 높이 증가와 접지점 하강이 동시에 멈춥니다. 현재는 이 상황을 감지하지 않습니다. 한계 항목에 정리했습니다.

## 6. 차량 그룹 단위 NMS

처음에는 클래스별로 NMS를 돌렸습니다. car는 car끼리, truck은 truck끼리, bus는 bus끼리 비교하는 방식입니다.

그런데 YOLO가 같은 차량을 car와 truck으로 동시에 검출하면, 서로 다른 클래스라서 박스가 두 개 다 남는 문제가 있었습니다. 추적기 쪽에서는 car, bus, truck을 전부 같은 사륜차 그룹으로 취급하고 있었으니, 검출 단계와 추적 단계의 기준이 어긋나 있었던 셈입니다.

그래서 NMS 기준도 추적 기준에 맞춰 그룹으로 묶었습니다.

- **사륜차 그룹**: car, bus, truck
- **이륜차 그룹**: bicycle, motorcycle
- **보행자 그룹**: person

```
검출 결과
→ NMS 그룹 분리
→ 그룹별 중복 제거 (IoU 0.45)
→ 최종 Detection 목록 생성
```

각 그룹별로 `NMSBoxes()`를 따로 돌린 뒤 결과를 하나의 Detection 목록으로 합쳤습니다.

**관찰 (미계측)**: 같은 차량에 서로 다른 클래스 박스가 겹쳐 남아 별도 Track이 생기는 경우가 눈에 띄게 줄었습니다. 정량 비교는 §31에 계측 항목으로 정리했습니다.

## 7. 포함 관계 중복 제거

그룹 NMS를 적용한 뒤에도 남는 중복이 있었습니다. 큰 박스가 작은 박스를 감싸고 있는데 IoU는 임계값에 못 미치는 경우입니다.

실제로 문제가 된 사례의 수치는 다음과 같았습니다.

```
기존 NMS         : IoU 약 0.36 < 0.45   → 중복 제거 실패
기존 containment : 약 0.70 < 0.85       → 가로로는 감싸지만 세로가 납작해서 실패
```

큰 박스가 여러 차량을 가로로 묶어 잡으면서 세로로만 납작해진 형태였습니다. 두 기준 모두 통과하지 못해 그대로 남았습니다.

다섯 가지를 수정했습니다.

1. exact class가 아니라 같은 NMS 그룹끼리 비교 (car / bus / truck 사이 분류 흔들림도 같은 차량 후보로 취급)
2. containment 0.60 이상이면 부분 포함 후보로 허용
3. 큰 박스 안에 작은 박스 중심이 들어 있고, 큰 박스 종횡비가 2.20 이상으로 비정상적으로 납작하면 부분 포함형 중복으로 판단
4. 면적 비율 기준을 1.60에서 1.35로 완화
5. 큰 박스의 형태가 비정상적이면 confidence가 0.08까지 더 높아도 큰 박스를 제거

```cpp
constexpr float strictContainment       = 0.85F;
constexpr float looseContainment        = 0.60F;
constexpr float minimumAreaRatio        = 1.35F;
constexpr float abnormalAspectRatio     = 2.20F;
constexpr float largeBoxConfidenceMargin = 0.08F;
```

이 필터는 차량 클래스에만 적용합니다. 보행자와 이륜차는 정상적인 상황에서도 박스가 서로 겹치기 때문에, 같은 규칙을 적용하면 실제 객체를 지우게 됩니다.

## 8. 극소 박스 제거

검출 하한선을 0.10까지 내리면서(§10) 노이즈성 초소형 박스가 늘었습니다. 높이가 몇 픽셀에 불과한 박스는 추적을 시작해도 프레임마다 크기가 심하게 흔들려 TTC-P 계산을 오염시킵니다.

```cpp
const int minimumBoxHeight =
    std::max(8, static_cast<int>(std::round(height * 0.014)));
```

1280×720 기준으로 10px입니다. 고정값 8px과 해상도 비율 1.4% 중 큰 값을 쓰기 때문에, 고해상도 입력에서도 같은 상대 크기가 유지됩니다.

## 9. 검출과 ROI 시각화 순서 조정

초기에는 ROI 오버레이를 프레임에 먼저 그린 다음 YOLO 추론을 돌렸습니다.

```
ROI 오버레이
→ YOLO 추론
```

이 경우 ROI 안쪽 픽셀 색이 초록색과 섞인 상태로 모델에 들어가서 정면 차량 검출 성능이 떨어졌습니다. 순서를 뒤집었습니다.

```
원본 프레임으로 YOLO 추론
→ Detection 생성
→ 출력용 프레임에 ROI 오버레이
→ 검출 결과 시각화
```

```cpp
detections = detectObjects(
    net,
    frame,
    detectorThreshold,
    nmsThreshold
);

// 추론이 끝난 뒤 시각화
cv::Mat overlay = frame.clone();
```

이후로는 추론 입력용 원본 프레임과 화면 표시용 프레임을 분리해서 관리했습니다.

같은 이유로 §29의 장면 전환 감지도 오버레이를 그리기 전의 원본 프레임을 씁니다. ROI 오버레이가 그려진 프레임끼리 비교하면 매 프레임 동일한 초록 영역이 차이 계산에 섞여 들어갑니다.

## 10. 검출 하한선 조정

초기 confidence threshold는 0.40이었습니다.

```cpp
constexpr float confidenceThreshold = 0.40F;
```

원거리 차량은 화면에서 차지하는 픽셀 수가 적고 특징도 약해 confidence가 낮게 나왔고, 이 기준에서 자주 걸러졌습니다. 우선 0.25로 낮췄습니다.

```cpp
constexpr float confidenceThreshold = 0.25F;
```

여기서 추적 안정성을 더 높이기 위해 검출 하한선을 0.10까지 내렸습니다.

```cpp
constexpr float detectorThreshold = 0.10F;
```

하한선을 내리면 오검출도 같이 늘어나므로, 낮은 confidence 후보를 어디까지 신뢰할지를 추적 단계에서 등급으로 나눠 처리했습니다. 자세한 내용은 §19에 정리했습니다.

## 11. 원거리 도로 영역 보조 추론

검출 하한선을 내려도 원거리 차량은 여전히 늦게 잡혔습니다. 640×640 letterbox 과정에서 원거리 차량이 몇 픽셀 수준까지 축소되기 때문입니다.

도로 소실점 부근을 crop해서 같은 모델로 한 번 더 추론하는 구조를 추가했습니다.

```cpp
const int cropX      = frame.cols * 0.25F;
const int cropY      = frame.rows * 0.38F;
const int cropWidth  = frame.cols * 0.50F;
const int cropHeight = frame.rows * 0.36F;
```

crop 결과에는 두 가지 제한을 걸었습니다.

```cpp
// crop은 원거리 차량 보조 검출용이므로 낮은 confidence 후보는 제외
if (detection.confidence < 0.20F) continue;

// person 등까지 중복 추론하면 오검출이 늘어나므로 차량 계열만 사용
if (!isVehicleClass(detection.classId)) continue;
```

crop 좌표를 원본 좌표로 복원한 뒤 전체 프레임 결과와 합치고, 병합된 목록에 2차 NMS를 다시 적용합니다. 두 추론이 같은 차량에 대해 조금씩 다른 박스를 내놓기 때문에, 병합만 하고 끝내면 중복이 그대로 남습니다.

```
전체 프레임 추론 (conf ≥ 0.10)
→ 원거리 crop 추론 (차량만, conf ≥ 0.20)
→ 좌표 복원 후 병합
→ 2차 NMS
→ 포함 관계 중복 제거
→ 극소 박스 제거
```

이 구조가 처리 시간에 미친 영향은 §13에서 계측했습니다.

## 12. ROI 내부 판정 방식

바운딩 박스가 ROI와 조금이라도 겹치면 내부로 판정하는 방식은 너무 느슨했습니다. 그래서 객체가 실제로 도로에 닿는 위치에 가까운 바운딩 박스 하단 중앙점을 접지점으로 썼습니다.

```cpp
const cv::Point groundPoint(
    box.x + box.width / 2,
    box.y + box.height
);
```

접지점이 넓은 도로 ROI 안에 있는지는 `pointPolygonTest()`로 확인했습니다.

```cpp
const bool insideRoad =
    cv::pointPolygonTest(
        roadRoi,
        groundPoint,
        false
    ) >= 0.0;
```

ego lane은 단순 내외부 판정이 아니라 차선 안에서의 좌우 위치도 필요하기 때문에, 사다리꼴의 위아래 변을 선형 보간해 정규화 x좌표를 함께 계산하는 별도 함수를 썼습니다.

```cpp
struct LanePosition {
    bool inside = false;
    float normalizedX = 0.5F;   // 0.0 = 차선 왼쪽 끝, 0.5 = 중앙, 1.0 = 오른쪽 끝
};
```

이 `normalizedX`는 §21의 LEAD 점수 계산과 §22의 추월 차량 판정에 쓰입니다.

최종 결과 영상의 박스 색상은 다음 네 단계로 구분했습니다.

- **주황**: 도로 ROI 밖
- **청록**: 도로 ROI 안, ego lane 밖 (SIDE)
- **파랑**: ego lane 안이지만 LEAD가 아님
- **초록 / 노랑 / 빨강**: LEAD (SAFE / CAUTION / DANGER)

## 13. 프레임별 추론 시간 측정과 병목 분석

YOLO 추론 시작과 종료 시각을 `std::chrono::steady_clock`으로 측정해 프레임별 지연시간을 기록했습니다.

초기에는 전체 검출 구간만 하나의 값으로 측정했지만, §11의 원거리 crop 보조 추론이 추가되면서 처리 시간이 크게 증가했습니다.

병목이 실제로 어디에서 발생하는지 확인하기 위해 측정 구간을 세 부분으로 나눴습니다.

```
전체 프레임 YOLO 추론
→ 원거리 crop YOLO 추론
→ Detection 병합 및 NMS 후처리
```

각 구간을 별도로 측정한 뒤 1,252프레임 전체의 평균을 계산했습니다.

| 측정 구간 | 평균 처리 시간 | 전체 추론시간 비중 |
| --- | ---: | ---: |
| 전체 프레임 YOLO | 209.62 ms | 50.18% |
| 원거리 crop YOLO | 208.09 ms | 49.81% |
| Detection 후처리 | 0.025 ms | 약 0.01% |
| **전체 검출 구간** | **417.73 ms** | **100%** |

현재 환경은 OpenCV DNN의 CPU 추론이며, 평균 검출 속도는 약 **2.39 FPS**였습니다. 입력 영상이 29.97fps이므로 실시간 대비 약 12.5배 느립니다.

측정 전에는 Detection 병합이나 그룹 NMS, 중복 박스 제거 같은 후처리도 처리시간 증가에 어느 정도 영향을 줄 것으로 예상했습니다. 하지만 실제 측정에서는 후처리가 평균 0.025ms에 불과했고, 전체 지연시간의 거의 전부가 두 번의 YOLO 추론에서 발생하고 있었습니다.

특히 원거리 crop은 원본 영상보다 작은 영역이지만, crop 이미지 역시 `detectObjects()`에 전달된 뒤 다시 640×640 입력으로 변환되어 동일한 YOLOv8n 네트워크 전체를 통과합니다. 그래서 전체 프레임 추론과 원거리 crop 추론의 평균 시간이 각각 209.62ms와 208.09ms로 거의 동일하게 측정됐습니다.

결과적으로 현재 구조는 사실상 다음과 같습니다.

```
YOLOv8n 640×640 CPU 추론
약 209.62 ms

+

원거리 crop
→ 640×640 변환
→ YOLOv8n CPU 추가 추론
약 208.09 ms

=

평균 417.73 ms
약 2.39 FPS
```

즉 원거리 작은 차량의 검출률을 높이기 위해 추가한 보조 추론이 검출 정확도 측면에서는 효과가 있었지만, 연산량 측면에서는 프레임당 YOLO 추론을 사실상 두 번 수행하게 만들어 처리 속도를 거의 절반으로 낮추는 비용을 발생시켰습니다.

이 측정을 통해 현재 파이프라인의 성능 병목은 추적, NMS, TTC-P 계산 같은 후처리가 아니라 **객체 검출 모델의 반복 추론**이라는 점을 확인했습니다.

현재 단계에서는 검출과 위험 판단 로직의 동작을 우선 검증하고, 이 값을 CPU 기준 Baseline으로 남겨두기로 했습니다. 이후 임베디드 환경으로 이식할 때 TensorRT FP16/INT8, GPU·NPU 가속, 보조 crop 추론 주기 조절, 프레임 스킵과 추적 보간 등을 적용해 이 Baseline과 비교할 계획입니다.

## 14. Kalman Filter 기반 다중 객체 추적

YOLO는 매 프레임을 독립적으로 검출하기 때문에, 지금 잡힌 차량이 이전 프레임의 차량과 같은 객체인지 알 방법이 없습니다. 이를 해결하기 위해 `MultiObjectTracker` 클래스를 따로 만들었습니다.

```
include/
├── Detection.hpp
├── MultiObjectTracker.hpp
└── RiskAnalyzer.hpp

src/
├── main.cpp
├── MultiObjectTracker.cpp
└── RiskAnalyzer.cpp
```

초기 Kalman Filter 상태는 6개(중심 x, 중심 y, x 방향 속도, y 방향 속도, 박스 너비, 박스 높이)였고, YOLO에서 직접 얻는 측정값은 중심 x, 중심 y, 박스 너비, 박스 높이 4개였습니다. Kalman Filter는 이전 위치와 속도로 다음 프레임 박스를 예측하고, 실제 YOLO Detection으로 그 예측값을 보정하는 구조입니다.

## 15. IoU와 중심 거리 기반 매칭

초기 추적기는 Kalman Filter가 예측한 박스와 현재 Detection 박스의 IoU만으로 동일 객체를 판단했습니다.

```
이전 Track 위치 예측
→ 현재 Detection과 IoU 계산
→ IoU가 높은 조합부터 매칭
→ 연결되지 않은 Detection에 새 ID 발급
```

문제는 원거리 차량이었습니다. 박스가 작으니 몇 픽셀만 이동해도 IoU가 크게 떨어졌습니다. IoU 하나로는 부족해서 다음 정보를 함께 봤습니다.

- 박스 IoU
- 중심점 거리
- 박스 대각선 크기에 따른 허용 이동 거리
- 박스 면적 변화
- 클래스 그룹
- 검출 신뢰도
- Track 누락 프레임 수

허용 이동 거리는 객체 크기에 따라 달라지도록 계산했습니다.

```cpp
return std::clamp(
    largestDiagonal * 0.85F + 8.0F,
    16.0F,
    320.0F
);
```

큰 차량은 프레임 사이에 더 많이 움직여도 허용하고, 작은 차량은 허용 거리를 좁게 유지하는 방식입니다.

판정 순서를 바꾼 것이 가장 큰 수정이었습니다. **IoU가 기준 이상이면 중심 거리와 면적 조건이 그 후보를 다시 거부하지 못하게** 했습니다. 거리와 면적 게이트는 IoU가 낮을 때만 보조 수단으로 동작합니다.

```cpp
const bool passedByIoU = iou >= iouGate;

if (!passedByIoU && centerDistance > distanceLimit) continue;
if (!passedByIoU && areaRatio > maximumAreaRatio)   continue;
```

게이트를 통과한 후보의 비용은 세 항의 가중합으로 계산합니다.

```cpp
float cost =
    (1.0F - iou)        * 0.58F +
    normalizedDistance  * 0.27F +
    sizeDifference      * 0.15F;
```

`normalizedDistance`는 `centerDistance / distanceLimit`을 2.0에서 자르고, `sizeDifference`는 `|log(areaRatio)| / log(maximumAreaRatio)`를 1.0에서 자릅니다. 두 항 모두 상한이 있어야 하나의 항이 비용 전체를 지배하지 않습니다.

같은 차량이 car와 truck 사이에서 분류가 바뀌는 것은 허용하되, 완전히 같은 classId 후보를 조금 우선하도록 페널티를 뒀습니다.

```cpp
if (track.classId != detection.classId) cost += 0.03F;
```

## 16. 초기 Track 완화 매칭

새로 생성된 Track은 아직 속도 정보가 없습니다.

```
새 Track 생성
→ x/y 속도 0
→ 다음 위치를 제자리로 예측
→ 실제 Detection은 이동
→ 매칭 실패
→ 다시 새 Track 생성
```

Kalman Filter가 속도를 학습하려면 최소 몇 프레임은 Detection이 이어져야 하는데, 기존 매칭 조건은 이미 정확한 속도를 알고 있다는 전제처럼 너무 엄격하게 짜여 있었습니다.

그래서 초기 Track에는 완화된 게이트를 적용했습니다. 초기 Track의 정의는 다음과 같습니다.

```cpp
const bool immatureTrack =
    !track.confirmed || track.hits < minConfirmationHits_;
```

매칭 단계별 게이트 값은 다음과 같습니다.

| 단계 | IoU 게이트 | 거리 배수 | 최대 면적비 |
| --- | ---: | ---: | ---: |
| 1차 고신뢰도 (일반) | 0.05 | 1.20 | 3.5 |
| 1차 고신뢰도 (초기 Track) | 0.01 | 2.50 | 5.0 |
| 2차 저신뢰도 (일반) | 0.03 | 1.10 | 3.0 |
| 2차 저신뢰도 (초기 Track) | 0.03 | 2.00 | 4.5 |
| 3차 복구 | 0.03 | 1.0 + (누락−1)×0.35, 최대 2.60 | 4.0 |

초기 Track에 거리 허용을 2.5배까지 열어줘서, 최소 2~3프레임 동안 속도를 학습할 기회를 줬습니다.

**관찰 (미계측)**: 같은 차량에 매 프레임 새 ID가 생기는 현상이 크게 줄었습니다. 정량 비교는 §31에 정리했습니다.

## 17. Tracking ID 가로채기 문제 개선

YOLO가 잠시 놓친 Track도 일정 프레임 동안은 목록에 남아 있습니다. 문제는 오래 놓친 Track과 직전 프레임까지 정상적으로 이어진 Track이 똑같은 우선순위로 현재 Detection을 두고 경쟁했다는 점입니다. 오래된 Track의 예측 위치가 우연히 현재 차량 근처로 오면, 정상 ID 대신 예전 ID가 Detection을 가져가 버렸습니다.

매칭 단계를 나눴습니다.

1. **1차 매칭** — `missedFrames == 1`인 Track + 고신뢰도 Detection
2. **2차 매칭** — 1차에서 연결되지 않은 최근 Track + 저신뢰도 Detection
3. **3차 복구 매칭** — `2 ≤ missedFrames ≤ 10`인 확정 Track + 남은 고신뢰도 Detection

단계를 나눈 것만으로는 부족해서, 3차 복구 단계에는 누락 프레임 수에 비례하는 비용 페널티를 추가했습니다.

```cpp
if (matchPass == MatchPass::RecoveryHigh) {
    cost += static_cast<float>(track.missedFrames - 1) * 0.08F;
}
```

10프레임 놓친 Track은 비용이 0.72 올라가므로, 사실상 다른 후보가 없을 때만 연결됩니다.

복구 가능한 누락 프레임 수도 제한했습니다.

```cpp
const int maximumRecoveryFrames = std::min(maxMissedFrames_, 10);
```

Track 보관 자체는 20프레임까지 하지만, 복구 매칭은 10프레임까지만 시도합니다. 그 이상 끊긴 Track은 위치 예측 신뢰도가 낮아 잘못 연결될 위험이 더 큽니다.

**관찰 (미계측)**: 정상적으로 유지되던 ID를 과거 ID가 가로채는 경우가 줄었습니다.

## 18. Kalman Filter 상태 확장

차량이 카메라 쪽으로 다가오면 중심 위치뿐 아니라 박스 너비와 높이도 계속 커집니다. 기존 Kalman Filter는 너비·높이 값 자체는 알고 있었지만, 그것이 얼마나 빠르게 변하는지는 상태에 포함하지 않았습니다.

상태값을 6개에서 8개로 늘렸습니다.

```
[0] 중심 x        [4] 박스 너비
[1] 중심 y        [5] 박스 높이
[2] x 방향 속도    [6] 너비 변화속도
[3] y 방향 속도    [7] 높이 변화속도
```

측정값은 그대로 4개(중심 x, 중심 y, 너비, 높이)입니다.

```cpp
kalman(8, 4, 0, CV_32F)
```

박스 크기 변화속도까지 상태에 넣어, 차량이 접근하거나 멀어질 때 다음 프레임의 크기 변화도 예측에 반영되도록 했습니다.

매칭이 성립하면 위치와 크기는 실제 YOLO 박스로 덮어쓰고, 속도와 크기 변화속도는 Kalman Filter가 추정한 값을 그대로 유지합니다. 측정값으로 직접 관측할 수 없는 상태만 필터가 계속 들고 가는 구조입니다.

## 19. 고·저신뢰도 2단계 Detection 활용

confidence가 0.41 → 0.28 → 0.18 → 0.33처럼 프레임마다 흔들리는 경우가 있었습니다. 0.25 하나만 기준으로 쓰면 0.18이 나온 프레임에서 Detection이 사라지고, 이것이 반복되면 Track이 삭제됩니다.

ByteTrack 방식을 참고해 §10에서 낮춘 하한선(0.10) 위로 Detection을 두 등급으로 나눠 썼습니다.

```cpp
MultiObjectTracker tracker(
    0.25F,  // 고신뢰도 기준
    0.10F,  // 저신뢰도 최저 기준
    3,      // ID 확정에 필요한 연속 고신뢰도 매칭 횟수
    20      // 확정 Track 보관 프레임
);
```

- **고신뢰도 Detection (confidence ≥ 0.25)**: 기존 Track 매칭에 사용
- **저신뢰도 Detection (0.10 ≤ confidence < 0.25)**: 기존 Track 유지에만 사용

신규 Track 생성은 두 기준보다 더 높은 별도 임계값을 씁니다.

```cpp
constexpr float newTrackConfidenceThreshold = 0.30F;
```

Track 유지는 관대하게, 새 객체 생성은 엄격하게 나눈 것입니다. 한 프레임짜리 오검출이 새 ID를 소모하는 것을 막기 위해서입니다.

ID 확정 조건에도 같은 원칙을 적용했습니다. 저신뢰도 매칭은 Track을 살려두지만 `consecutiveHits`를 올리지 않기 때문에, 저신뢰도 검출만으로는 미확정 Track이 정식 ID로 승격되지 않습니다.

```cpp
const bool lowConfidenceMatch = matchPass == MatchPass::RecentLow;

if (!lowConfidenceMatch) ++track.consecutiveHits;

if (!track.confirmed &&
    !lowConfidenceMatch &&
    track.consecutiveHits >= minConfirmationHits_) {
    track.confirmed = true;
    track.id = nextTrackId_;
    ++nextTrackId_;
}
```

Track 삭제 기준도 확정 여부에 따라 다릅니다.

```cpp
// 미확정 Track: 2프레임 연속 놓치면 삭제
// 확정 Track  : maxMissedFrames_(20)까지 보관
if (!track.confirmed) return track.missedFrames > 1;
return track.missedFrames > maxMissedFrames_;
```

문서와 다이어그램에 적힌 "Track 최대 보관 20프레임"은 확정 Track에만 해당합니다.

## 20. 위험 분석 영역을 ego lane으로 분리

초기에는 넓은 driving ROI 안의 모든 객체를 위험 대상으로 분석했습니다. 그런데 옆 차선 차량이 카메라 가까이 지나가면 화면에서 박스가 빠르게 커지기 때문에, 실제 충돌 가능성과 무관하게 위험으로 오판할 수 있었습니다.

ROI를 두 개로 나눴습니다.

```
roadRoi     → 넓은 도로 관심 영역, 객체 카운트와 색상 구분용
egoLaneRoi  → 실제 선행 차량을 선택할 자차 차선 영역
```

좌표는 §3에 정리했습니다. 1280×720 기준으로 `roadRoi` 하단 폭은 1179px(화면 폭의 92%)이고, `egoLaneRoi` 하단 폭은 746px(58%), 상단 폭은 102px(8%)입니다.

옆 차선 차량은 계속 검출하고 추적하되, 위험 분석 대상에서만 제외했습니다. `roadRoi` 안이지만 `egoLaneRoi` 밖이면 SIDE 객체로 분류합니다.

**관찰 (미계측)**: 측면의 대형 트럭이나 인접 차선 차량이 위험 경고를 일으키는 문제가 줄었습니다.

## 21. 선행 차량 LEAD 선택

ego lane 안에 차량이 여러 대 있을 수 있어서, 위험 분석 대상 한 대를 고르는 로직을 추가했습니다.

점수는 접지점 y좌표에서 차선 중앙 이탈 페널티를 빼는 방식으로 계산합니다.

```cpp
const float centerPenalty =
    std::abs(lanePosition.normalizedX - 0.5F) * 90.0F;

leadScore = static_cast<float>(groundPoint.y) - centerPenalty;
```

화면 아래쪽에 있을수록(가까울수록) 점수가 높고, 차선 중앙에서 벗어날수록 감점됩니다. 페널티 계수 90은 차선 가장자리에 있는 차량이 중앙 차량보다 약 45px 아래에 있어야 앞설 수 있다는 뜻입니다.

프레임마다 후보가 조금씩 흔들려 LEAD ID가 계속 바뀌는 것을 막기 위해 히스테리시스를 넣었습니다.

```cpp
if (!riskAnalysisEnabled) {
    activeLeadId = -1;
} else if (!activeLeadVisible) {
    activeLeadId = proposedLeadId;
} else if (proposedLeadId >= 0 &&
           proposedLeadId != activeLeadId &&
           proposedLeadScore > activeLeadIterator->second + 35.0F) {
    activeLeadId = proposedLeadId;
}
```

기존 LEAD가 계속 보이면 유지하고, 새 후보는 35점 이상 높을 때만 교체합니다. 위험 분석 대상이 한 대로 좁혀지면서 화면 정보도 단순해지고 경고 기준도 더 명확해졌습니다.

## 22. LEAD 안정화: 체류 조건과 추월 차량 제외

§21의 점수 히스테리시스만으로는 두 가지 상황을 막지 못했습니다.

### 22-1. 길가 트럭의 순간 LEAD 선정

테스트 영상 35초 부근에서, 길가에 정차한 트럭의 접지점이 잠깐 ego lane 사다리꼴에 들어오면서 곧바로 LEAD가 되는 문제가 있었습니다. 접지점이 화면 아래쪽에 있어 점수가 높게 나왔기 때문입니다.

ego lane 안에 일정 시간 연속으로 머문 차량만 후보가 될 수 있도록 조건을 추가했습니다.

```cpp
const int leadEligibilityFrames =
    std::max(1, static_cast<int>(std::round(sourceFps * 0.5)));

std::unordered_map<int, int> egoLaneStreakById;

if (lanePosition.inside) ++laneStreak;
else                     laneStreak = 0;

if (lanePosition.inside &&
    isVehicleClass(trackedObject.classId) &&
    laneStreak >= leadEligibilityFrames) {
    // LEAD 후보 등록
}
```

29.97fps 기준 15프레임입니다.

### 22-2. 추월 차량의 DANGER 오경고

같은 35초 구간에서, 자차를 추월하며 옆으로 빠지는 차량이 순간적으로 ego lane을 통과하면서 DANGER가 뜨는 문제가 있었습니다. 추월 차량은 카메라에 가까워 박스가 빠르게 커지므로 TTC-P가 작게 계산됩니다.

차선 중앙에서 바깥쪽으로 밀려나는 속도를 계산해 추월 차량을 구분했습니다.

```cpp
constexpr std::size_t lateralHistorySize = 8;
constexpr float maximumLateralDriftPerFrame = 0.02F;

// 차선 중앙(0.5)으로부터의 이탈 정도 변화율
const float pastOffset   = std::abs(lateralHistory.front() - 0.5F);
const float recentOffset = std::abs(lateralHistory.back()  - 0.5F);

outwardDriftPerFrame =
    (recentOffset - pastOffset) / (lateralHistory.size() - 1);

const bool passingBy = outwardDriftPerFrame > maximumLateralDriftPerFrame;
```

이력이 4개 이상 쌓여야 계산하고, 최대 8개를 유지합니다. `passingBy`로 판정된 차량은 경고 대상에서 제외합니다.

이 판정에는 구조적 비대칭이 있습니다. `outwardDrift`가 **음수**이면 차선 중앙으로 파고드는 차량, 즉 컷인입니다. 현재는 그 부호를 사용하지 않고 있어서, 컷인 차량도 §22-1의 0.5초 체류 조건을 그대로 기다려야 합니다. 개선 항목에 정리했습니다.

## 23. TTC-P 설계

단안 블랙박스 영상만으로는 차량까지의 실제 거리나 실제 상대속도를 정확히 알 수 없습니다. 카메라 내부 파라미터, 차량 실제 크기, 도로 기하 정보가 없기 때문에, 결과를 실제 TTC라고 부르지 않고 `TTC-P(TTC Proxy)`라는 이름으로 구분했습니다.

전방 물체의 화면상 높이 `h`는 실제 거리 `Z`에 대략 반비례합니다.

```
h ≈ k / Z
log(h) ≈ log(k) − log(Z)
d(log h)/dt ≈ 1 / TTC
```

따라서 `TTC-P ≈ 1 / (log 높이 증가율)`이 됩니다. 카메라 보정도 자차 속도도 필요 없이 박스 높이 시계열만으로 계산할 수 있습니다.

한두 프레임의 박스 흔들림에 결과가 휘둘리지 않도록, 첫 샘플과 끝 샘플의 차분이 아니라 저장된 전체 샘플에 최소제곱 선형회귀를 적용했습니다.

```cpp
// 기울기 = Σ((t−t̄)(v−v̄)) / Σ((t−t̄)²)
return static_cast<float>(numerator / denominator);
```

`RiskAnalyzer`의 주요 파라미터는 다음과 같습니다.

```cpp
RiskAnalyzer riskAnalyzer(sourceFps, 15, 60);
//                                  │   └ staleFrameLimit: 60프레임 미검출 시 이력 삭제
//                                  └ historySize: 최대 15샘플 보관

maximumHistoryGapFrames_ = 5;   // 5프레임 넘게 끊기면 이전 이력과 이어붙이지 않음
```

`maximumHistoryGapFrames_`는 같은 ID가 오래 끊겼다가 다시 잡혔을 때 박스 크기가 크게 달라져 TTC-P가 튀는 것을 막기 위한 조건입니다.

샘플이 8개 미만이면 박스 흔들림인지 실제 확대 추세인지 구분할 수 없어 TTC-P를 계산하지 않습니다.

```cpp
if (history.samples.size() < 8) return result;
```

박스 높이가 거의 변하지 않는 구간에서 노이즈성 TTC-P가 튀는 것을 막기 위해, 증가율이 일정 이상일 때만 계산합니다.

```cpp
if (logHeightRate > 0.02F) {
    ttcSeconds = 1.0F / logHeightRate;
}
```

조건을 만족하지 못하면 TTC-P는 무한대로 두고 화면에는 `--`로 표시됩니다.

너비가 아니라 높이를 스케일 지표로 쓴 이유는, 차선 변경이나 부분 가림에서 너비가 심하게 흔들리는 데 비해 높이는 상대적으로 안정적이기 때문입니다.

## 24. 위험 단계 시간 안정화

검출 박스는 매 프레임 조금씩 흔들리기 때문에 TTC-P도, 위험 단계도 순간적으로 오르내릴 수 있습니다.

```
SAFE → CAUTION → SAFE → CAUTION
```

이 상태를 그대로 화면에 내보내면 경고가 계속 깜빡입니다. `RiskAnalyzer::stabilizeLevel()`에 승격과 강등을 비대칭으로 처리하는 로직을 넣었습니다.

```cpp
// 승격: DANGER는 빨리 반응해야 하므로 2프레임, CAUTION은 오경고를 줄이려고 4프레임
const int requiredFrames = rawLevel == RiskLevel::Danger ? 2 : 4;

// 강등: DANGER였으면 10프레임, CAUTION이었으면 6프레임 동안 낮은 값이 유지돼야 내려감
const int requiredFrames = history.stableLevel == RiskLevel::Danger ? 10 : 6;
```

상단 배너는 이 안정화 결과 위에 한 층 더 확인 조건을 겁니다.

```cpp
constexpr int cautionConfirmationFrames = 8;
constexpr int dangerConfirmationFrames  = 3;

const int warningHoldFrames =
    std::max(1, static_cast<int>(std::round(sourceFps * 0.25)));
```

두 층이 직렬이므로 실효 지연은 다음과 같습니다.

| 경로 | 내부 안정화 | 배너 확인 | 합계 (29.97fps) |
| --- | ---: | ---: | ---: |
| SAFE → CAUTION 배너 | 4프레임 | 8프레임 | 12프레임 (0.40초) |
| SAFE → DANGER 배너 | 2프레임 | 3프레임 | 5프레임 (0.17초) |

배너 유지 시간은 처음 0.4초로 뒀다가, 정상 상태로 돌아온 뒤 오래 남아 보이는 문제가 있어 0.25초로 줄였습니다.

Track이 LEAD에서 빠지면 그 Track의 위험 이력은 초기화합니다.

```cpp
if (!isLeadTarget) {
    resetTrackHistory(history);
    return RiskResult{};
}
```

이 초기화에는 부작용이 있습니다. 접지점이 ego lane 사다리꼴 경계를 한 프레임만 벗어나도 15샘플이 전부 삭제되고, `laneStreak`도 0으로 돌아갑니다. ego lane 상단 폭이 화면 폭의 8%(1280 기준 102px)에 불과해 원거리에서는 이 경계를 자주 넘나듭니다. 개선 항목에 정리했습니다.

## 25. 경고 단계와 TTC-P 숫자 표시 분리

위험 단계는 §24의 안정화 때문에 잠시 유지되는데, TTC-P 숫자는 매 프레임 갱신됩니다. 이 때문에 다음처럼 모순되어 보이는 표시가 나왔습니다.

```
CAUTION: CLOSING LEAD 17.5s
```

이전 프레임에서 TTC-P가 낮아 CAUTION이 확정된 뒤, 다음 프레임에서 TTC-P는 올라갔지만 위험 단계는 강등 조건(6프레임) 때문에 아직 유지되고 있던 상황이었습니다.

여기서 두 가지 선택지가 있었습니다.

1. 안정화 로직을 약화시켜 단계와 숫자를 동기화한다
2. 두 정보의 표시 위치를 나눠서 각자 다른 역할을 맡긴다

1번을 택하면 §24에서 해결한 깜빡임 문제가 되돌아옵니다. 이 모순은 안정화 로직의 버그가 아니라 **의도한 동작이 화면에 그대로 노출된 것**이므로, 로직이 아니라 표시 방식을 고치는 것이 맞다고 판단했습니다.

상단 배너에서는 TTC-P 숫자를 뺐습니다.

```
CAUTION: CLOSING LEAD
DANGER:  LOW TTC-P
```

역할을 다음과 같이 나눴습니다.

- **상단 배너**: 현재 확정된 위험 단계 (시간 안정화된 값)
- **LEAD 라벨 / 좌측 상태창**: 현재 프레임의 TTC-P 수치 (실시간 값)

배너는 "지금 어떤 상태인가", 라벨은 "지금 무슨 값이 관측되고 있는가"를 각각 맡게 되면서, 두 정보가 서로 모순처럼 보이는 문제는 사라졌습니다. 안정화 강도는 그대로 유지했습니다.

## 26. 우측 탱크로리 원시 후보 진단

19초부터 화면 오른쪽에 탱크로리가 보이는데, 22초가 지나야 truck 박스가 생기는 문제가 있었습니다.

처음에는 입력 해상도 문제라고 보고 우측 영역을 crop한 뒤 추가 추론하는 방식도 시험해봤습니다.

```
전체 프레임 YOLO 추론
→ 우측 영역 crop
→ crop을 640×640으로 확대
→ 추가 YOLO 추론
→ 원본 좌표로 복원
```

그런데 처리 시간만 늘고 검출 시점은 거의 나아지지 않았습니다. **가설이 틀렸다는 뜻이었습니다.**

원인을 정확히 확인하기 위해 18~24초 구간의 필터링 전 YOLO 후보를 CSV로 저장했습니다. frame, time, 상위 1·2·3순위 클래스, car / bus / train / truck / boat 점수, 박스 좌표와 크기, confidence 통과 여부, 대상 클래스 통과 여부, 최종 필터 통과 여부까지 기록했습니다. (`results/yolo_raw_right_edge_diagnostic.csv`)

결과를 보니 문제의 큰 박스는 21.9초부터 이미 생성되고 있었는데, 클래스가 계속 train(6)으로 분류되고 있었습니다.

```
21.922초  train 0.768
21.989초  train 0.886
22.055초  train 0.904
22.122초  train 0.914
22.289초  truck 0.607
```

즉 검출 위치를 찾지 못한 것이 아니라, 클래스 오분류와 후처리 필터의 문제였습니다. 원시 후보를 덤프하지 않았다면 계속 해상도 쪽을 파고들었을 것입니다.

## 27. train으로 오분류된 탱크로리 제한적 복구

기존 검출 코드는 최고 점수 클래스가 ADAS 대상이 아니면 후보를 바로 삭제했습니다.

```
탱크로리 박스 생성
→ 최고 클래스 train
→ isTargetClass(6) == false
→ Detection 삭제
→ Tracker에 전달되지 않음
```

모든 train을 truck으로 바꾸면 다른 장면에서 오검출이 늘어날 위험이 있어 조건을 강하게 제한했습니다.

```cpp
const bool isRightTankerMisclassifiedAsTrain =
    rawClassId == 6 &&
    rawConfidence >= 0.20F &&
    left            >= static_cast<int>(frame.cols * 0.45F) &&
    restoredCenterX >= static_cast<int>(frame.cols * 0.60F) &&
    right           >= static_cast<int>(frame.cols * 0.76F) &&
    restoredWidth   >= static_cast<int>(frame.cols * 0.18F) &&
    restoredHeight  >= static_cast<int>(frame.rows * 0.20F) &&
    restoredArea    >= static_cast<int>(
        static_cast<float>(frame.cols * frame.rows) * 0.055F) &&
    bottom          >= static_cast<int>(frame.rows * 0.62F);
```

7개 조건 AND입니다. 진단 CSV에서 확인한 실제 박스보다 약간 여유 있게 잡아, 프레임별 박스 흔들림 때문에 조건이 끊기는 것을 줄였습니다.

복구된 박스의 표시 신뢰도는 원본보다 낮춰서, 정상 검출과 구분되도록 했습니다.

```cpp
const float displayConfidence =
    isRightTankerMisclassifiedAsTrain
        ? std::clamp(rawConfidence * 0.55F, 0.25F, 0.49F)
        : rawConfidence;
```

NMS에는 원본 점수(`nmsScores`)를 쓰고, 화면 표시와 Track 신뢰도에는 낮춘 점수를 씁니다. 복구된 후보가 중복 제거 단계에서 불리해지지 않으면서도, 결과 영상에서는 추정 결과임이 드러나도록 나눈 것입니다.

이 방식으로 21.9초 이후의 큰 탱크로리 박스는 기존 파이프라인 안에서 계속 추적할 수 있게 됐습니다. 다만 19~21초 초반에는 유효한 후보 자체가 거의 없었기 때문에, 이 구간은 현재 모델의 한계로 남겨뒀습니다.

**이 조건은 현재 테스트 영상 전용입니다.** 좌표, 면적, 위치가 모두 특정 장면에서 역산한 값이라 다른 영상에는 적용되지 않습니다. 근본 해결은 대형 차량 파인튜닝입니다.

## 28. 원거리 작은 차량 오경고 필터링

멀리 있는 작은 선행 차량은 바운딩 박스 높이가 몇 픽셀만 바뀌어도 비율상 큰 변화로 계산됩니다. 테스트 영상 14.98~15.58초 구간에서, 원거리 선행 차량의 박스가 2~3픽셀 흔들리면서 TTC-P가 약 4초로 계산되어 CAUTION이 뜨는 문제가 있었습니다.

TTC-P 임계값 하나만 보지 않고 여러 조건을 함께 확인하도록 했습니다.

```cpp
const int minimumLeadBoxHeightForWarning =
    std::max(36, static_cast<int>(std::round(height * 0.05)));

const int minimumLeadGroundYForWarning =
    static_cast<int>(std::round(height * 0.60));

constexpr int   minimumRiskSamplesForWarning       = 10;
constexpr float minimumHeightGrowthRatioForWarning = 1.08F;
constexpr float minimumGroundSpeedForWarning       = 2.0F;
```

```
원시 RiskAnalyzer 결과
→ 경고 신뢰성 조건 검사
→ 조건 부족 시 화면 경고 차단
```

조건을 통과하지 못한 위험 결과는 내부 계산값이 있더라도 화면에서는 SAFE로 처리합니다.

### 현재 구조의 문제: 임계값이 두 층에 존재

이 필터는 `main.cpp`에 **추가로** 붙였기 때문에, `RiskAnalyzer` 내부에 원래 있던 판정 조건과 직렬로 걸립니다. 두 층을 나란히 놓으면 다음과 같습니다.

| 조건 | RiskAnalyzer 내부 | main.cpp 필터 | 실효값 |
| --- | --- | --- | --- |
| 박스 높이 | DANGER 18px / CAUTION 12px | ≥ max(36, 5%H) | **36px** |
| 높이 증가율 | DANGER 1.06 / CAUTION 1.025 | ≥ 1.08 | **1.08** |
| 접지점 속도 | DANGER 8.0 / CAUTION 2.5 | ≥ 2.0 | **2.5 / 8.0** |
| TTC-P | DANGER ≤2.5초 / CAUTION ≤5.0초 | — | **2.5 / 5.0초** |
| 샘플 수 | ≥ 8 (TTC-P 산출) | ≥ 10 (경고 허용) | **8 / 10** |

직렬이므로 실효값은 두 값 중 엄격한 쪽입니다. 그 결과 **내부의 12px, 18px, 1.025, 1.06과 바깥의 2.0px/s는 어떤 입력에서도 판정에 관여하지 않습니다.** CAUTION과 DANGER가 크기·증가율 조건을 공유하게 되어, 실제로 두 단계를 가르는 것은 접지점 속도와 TTC-P 두 개뿐입니다.

원인은 오경고를 `RiskAnalyzer`의 조건을 조정해서 잡지 않고 바깥에 새 필터를 덧붙여 잡았기 때문입니다. 정리 방향은 개선 항목에 적었습니다.

## 29. 장면 전환 감지와 상태 초기화

입력 영상은 하나의 연속 주행 영상이 아니라 여러 도로 장면이 편집되어 이어진 영상입니다. 이전 장면의 경고 유지 카운터와 Track이 다음 장면까지 그대로 남는 문제가 있었습니다.

### 29-1. 1차 구현: 2조건

처음에는 두 가지만 봤습니다.

```
프레임을 320×180 회색 영상으로 축소
→ 이전 프레임과 평균 절대 차이 계산
→ 밝기 히스토그램 상관계수 계산
→ 둘 다 임계값을 넘으면 장면 전환 판단
```

이 구조에서는 **13초와 17초의 햇빛/자동 노출 변화가 장면 전환으로 오검출**됐습니다. 역광 구간에 진입하면 화면 전체 밝기가 급변해 평균 절대 차이가 커지고 히스토그램 상관도도 떨어지는데, 실제로는 같은 장면입니다. 이때마다 Track과 위험 이력이 초기화되어 추적이 끊겼습니다.

### 29-2. 2차 구현: 밝기 이동 성분 분리

노출 변화와 실제 컷의 차이는 **변화가 화면 전체에 균일한가**입니다. 노출이 바뀌면 모든 픽셀이 같은 방향으로 이동하고, 컷이 나면 픽셀마다 제각각 바뀝니다.

전체 변화량 중 전역 밝기 이동이 차지하는 비율을 계산했습니다.

```cpp
const float meanShift = currentMean - previousMean;

const float shiftRatio =
    std::abs(meanShift) / std::max(frameDifference, 1e-3F);
```

그리고 전역 밝기 이동을 빼고 남는 구조 변화량도 따로 계산했습니다.

```cpp
cv::Mat compensatedDifferenceImage = currentFloat - previousFloat;
compensatedDifferenceImage -= meanShift;

const float compensatedDifference =
    static_cast<float>(cv::mean(compensatedAbsolute)[0]);
```

동영상 전체 프레임에서 두 유형의 `shiftRatio`를 측정한 결과는 다음과 같았습니다.

| 유형 | shiftRatio 범위 |
| --- | --- |
| 실제 컷 | 약 0.11 ~ 0.26 |
| 햇빛 / 자동 노출 변화 | 약 0.59 ~ 0.93 |

두 분포가 겹치지 않아, 0.50을 기준으로 깨끗하게 나눌 수 있었습니다.

최종 판정은 4개 조건 AND입니다.

```cpp
constexpr float diffThreshold             = 20.0F;
constexpr float histogramThreshold        = 0.78F;
constexpr float maximumExposureShiftRatio = 0.50F;
constexpr float compensatedDiffThreshold  = diffThreshold * 0.80F;  // 16.0

const bool sceneChanged =
    frameDifference       >= diffThreshold        &&
    correlation           <= histogramThreshold   &&
    shiftRatio            <  maximumExposureShiftRatio &&
    compensatedDifference >= compensatedDiffThreshold;
```

`shiftRatio`로 걸러진 경우는 별도 로그로 남겨서, 오검출을 막은 것인지 실제 컷을 놓친 것인지 확인할 수 있게 했습니다.

```
[SCENE CHANGE]    frame=... diff=... histogram=... shiftRatio=... compensatedDiff=...
[EXPOSURE CHANGE] frame=... diff=... histogram=... shiftRatio=... compensatedDiff=...
```

**계측 결과**: 1,252프레임 중 장면 전환 판정 3회(프레임 322 / 574 / 1010). 세 지점 모두 육안으로 확인한 하드컷이며, 역광 구간에서의 오검출은 발생하지 않았습니다.

### 29-3. 전환 시 초기화 범위

장면 전환이 감지되면 다음을 초기화합니다.

```cpp
if (sceneChanged) {
    cautionHoldRemaining   = 0;
    dangerHoldRemaining    = 0;
    cautionCandidateFrames = 0;
    dangerCandidateFrames  = 0;
    warningCandidateLeadId = -1;
    activeLeadId           = -1;
    sceneWarmupRemaining   = sceneWarmupFrames;   // 0.5초

    egoLaneStreakById.clear();
    lateralHistoryById.clear();
}
```

새 장면 직후 0.5초 동안은 모든 객체를 `isLeadTarget = false`로 전달해 위험 분석을 멈춥니다. 검출과 추적이 안정화된 뒤에 LEAD를 다시 고르기 위해서입니다.

**현재 `MultiObjectTracker`는 초기화하지 않습니다.** 리셋 함수가 구현되어 있지 않아, 이전 장면의 Kalman 상태와 Track ID가 새 장면으로 이월됩니다. 실제로 프레임 574의 컷에서 ID:1 트랙이 컷을 넘어 다른 장면의 다른 차량에 연결되는 것을 확인했습니다. TTC-P 이력은 warmup 구간에서 초기화되므로 위험 판정 자체는 오염되지 않지만, ID의 의미가 깨집니다. 개선 항목에 정리했습니다.

## 30. 한국어 UI와 결과 화면 정리

OpenCV 기본 Hershey 글꼴은 한글을 지원하지 않아, OpenCV FreeType과 시스템 한글 글꼴(NanumGothic 또는 Noto Sans CJK)을 함께 썼습니다.

최종 결과 화면은 다음 기준으로 정리했습니다.

- **객체 클래스명**: car, truck, person 등 영어 유지
- **왼쪽 상태 정보**: 한국어 (프레임 번호, 추론 시간, 도로 객체 수, 내 차선 객체 수, 선행 차량 상태)
- **상단 경고 배너**: 한국어
- **진단용 원시 후보**: 최종 영상에서 제거
- **핑크색 디버그 ROI**: 제거
- **왼쪽 아래 설명 문구**: 제거

디버깅 버전과 포트폴리오용 결과 버전을 분리해서, 화면에는 필요한 정보만 남겼습니다.

LEAD 박스와 라벨은 다른 객체를 모두 그린 뒤 마지막에 다시 그립니다. 큰 트럭 박스가 LEAD 라벨을 덮는 문제가 있었기 때문입니다.

---

## 31. 결과 영상 기반 사후 계측

§6, §16, §17, §20, §28의 개선 효과는 지금까지 "줄었습니다"로만 기록했고 수치가 없었습니다. §13에서 처리 시간을 계측한 것과 대비하면, **재기 쉬운 지표만 재고 정확도 관련 지표는 재지 않은 상태**였습니다.

정답 라벨 없이도 결과 영상의 오버레이 정보만으로 계측 가능한 항목이 있어, 현재 버전(`step7_scene_change_fixed_output.avi`, 1,252프레임 / 41.8초)을 대상으로 우선 측정했습니다.

| 항목 | 계측값 |
| --- | ---: |
| 총 프레임 | 1,252 (41.8초 @ 29.97fps) |
| 프레임당 평균 추론 시간 | 417.73 ms |
| 처리 속도 | 2.39 FPS |
| LEAD가 지정된 프레임 비율 | 48.8% |
| TTC-P 숫자가 산출된 프레임 비율 | 16.7% (LEAD 프레임 중 34.2%) |
| LEAD 유지 구간 수 | 16개 |
| LEAD 유지 구간 길이 (중앙값 / 최대) | 3프레임 / 237프레임 |
| 15프레임 미만으로 끊긴 LEAD 구간 | 16개 중 12개 |
| TTC-P 연속 산출 구간 수 | 40개 |
| TTC-P 연속 구간 길이 (중앙값 / 최대) | 3프레임 / 42프레임 |
| 10프레임 이상 이어진 TTC-P 구간 | 5개 |
| 장면 전환 판정 | 3회 |
| CAUTION / DANGER 배너 발생 | 0회 |
| 최대 Track ID | 62 |

이 표에서 확인된 것은 두 가지입니다.

**첫째, LEAD가 거의 유지되지 않습니다.** §22-1에서 LEAD 안정화를 위해 0.5초(15프레임) 체류 조건을 넣었는데, 정작 지정된 LEAD의 75%가 15프레임을 채우지 못하고 끊깁니다. 체류 조건은 진입만 막고 유지는 보장하지 않기 때문입니다. §24에서 언급한 `resetTrackHistory` 초기화가 원인으로 보입니다.

**둘째, 이 영상으로는 경고 경로를 검증할 수 없습니다.** 41.8초 동안 CAUTION과 DANGER가 한 번도 발생하지 않았습니다. 위험 상황이 포함되지 않은 영상이기 때문입니다. 따라서 §28의 신뢰성 조건, §24의 배너 확인 프레임 수, 배너 유지 시간은 **실행된 적이 없는 상태로 값만 정해져 있습니다.**

이 계측은 결과 영상의 오버레이를 파싱해서 얻은 것이라 재현이 번거롭습니다. 다음 버전에서는 프로그램 내부에 카운터를 넣어 종료 시 함께 출력하도록 할 계획입니다.

```
- 프레임당 신규 ID 발급 수
- 확정 전 소멸한 Track 수 / 확정된 Track 수
- 3차 복구 매칭 성공 횟수
- LEAD 교체 횟수, LEAD 유지 구간 길이 분포
- TTC-P 산출 성공률
- CAUTION / DANGER 발생 횟수 및 지속 프레임 수
```

이 값들이 있으면 §6, §16, §17의 "줄었습니다"를 `이전 N회 → 이후 M회` 형태로 다시 쓸 수 있습니다.

---

## 현재 구현 상태

```
블랙박스 영상 입력
→ 영상 정보 확인 (해상도, FPS)
→ 장면 전환 감지 (원본 프레임 기준, 4조건)
→ 전환 시 경고/LEAD/체류/횡이동 상태 초기화 + 0.5초 분석 유예
→ Letterbox 전처리
→ YOLOv8 ONNX 전체 프레임 추론 (conf ≥ 0.10)
→ 원거리 도로 crop 보조 추론 (차량만, conf ≥ 0.20)
→ ADAS 클래스 필터링
→ 우측 대형 train 오분류 제한적 복구
→ 차량 그룹 단위 NMS (IoU 0.45)
→ 병합 후 2차 NMS
→ 포함 관계 중복 제거
→ 극소 박스 제거 (max(8px, 1.4%H))
→ Kalman Filter 위치·크기·변화속도 예측 (상태 8개)
→ 3단계 Track 매칭 (고신뢰도 → 저신뢰도 → 복구)
→ 객체별 ID 부여 (신규 생성 conf ≥ 0.30, 확정 3회)
→ roadRoi 접지점 판정
→ egoLaneRoi 내부 판정 + 정규화 x좌표 계산
→ ego lane 0.5초 체류 차량 중 최고 점수를 LEAD로 선택
→ 추월 차량(passing-by) 제외
→ 박스 높이·접지점 변화 이력 저장 (최대 15샘플)
→ TTC-P 계산 (8샘플 이상)
→ 위험 단계 안정화 (승격 4/2, 강등 6/10)
→ 경고 신뢰성 조건 검사
→ 배너 확인 (CAUTION 8프레임, DANGER 3프레임) 및 0.25초 유지
→ 한국어 상태 및 경고 출력
→ 결과 영상 저장
→ 구간별 Latency와 FPS 출력
```

## 현재 결과

**계측된 것**

- 평균 검출 시간 417.73ms, 2.39 FPS (§13)
- 병목의 99.99%가 두 번의 YOLO 추론 (§13)
- 장면 전환 판정 3회, 역광 구간 오검출 0회 (§29)
- 실제 컷과 노출 변화의 shiftRatio 분포 분리 (§29)
- 탱크로리 미검출의 원인이 위치 탐지 실패가 아닌 클래스 오분류임을 원시 후보 로그로 확인 (§26)
- LEAD 지정률 48.8%, TTC-P 산출률 16.7% (§31)

**정성 관찰 (미계측)**

- 연속적으로 검출되는 차량에 대해 동일 ID를 유지
- confidence가 잠깐 낮아져도 저신뢰도 Detection으로 Track 연결
- 오래 놓친 Track이 현재 차량 ID를 가로채는 현상 감소
- 같은 차량의 car·truck 중복 박스 감소
- 원거리 작은 차량의 순간적인 오경고 감소
- 측면 차량을 위험 분석에서 제외

**구조적으로 확보한 것**

- 넓은 도로 ROI와 실제 위험 분석용 ego lane 분리
- 자차 차선 선행 차량 한 대를 LEAD로 선택
- 영상 스케일 변화 기반 TTC-P 계산
- 한국어 상태 정보와 경고를 포함한 결과 영상 생성

## 남아 있는 한계

**1. 실제 거리 기반 TTC가 아님**

현재 TTC-P는 바운딩 박스 높이 증가율을 이용한 Proxy입니다. 정확한 거리 기반 TTC를 계산하려면 카메라 내부 파라미터, 카메라 설치 높이와 각도, 차량 실제 크기 추정, 차선과 도로 기하 정보, 카메라 자체 움직임 보정이 필요합니다.

**2. 박스 하단 절단 시 TTC-P 산출 불가**

§5의 좌표 clamp 때문에, 선행 차량이 매우 가까워져 박스 하단이 화면 밖으로 나가면 박스 높이 증가와 접지점 하강이 동시에 멈춥니다. `logHeightRate`가 0으로 수렴해 TTC-P가 무한대가 되고, `groundSpeed`도 0이 되어 §28의 경고 조건을 통과하지 못합니다. **위험이 가장 큰 시점에 경보가 SAFE로 되돌아가는 구조입니다.** 현재 테스트 영상에는 이 구간이 없어 실제로 관측되지는 않았습니다.

**3. 경고 경로가 검증되지 않음**

§31에서 확인한 대로, 현재 테스트 영상에서 CAUTION과 DANGER가 한 번도 발생하지 않았습니다. §24의 배너 확인 프레임 수와 유지 시간, §28의 신뢰성 조건은 실행된 적 없이 값만 정해진 상태입니다.

**4. 경고 임계값이 두 층에 중복 존재**

§28에 정리한 대로, `RiskAnalyzer` 내부 조건과 `main.cpp` 필터가 직렬로 걸리면서 5개 상수가 어떤 입력에서도 판정에 관여하지 않습니다. CAUTION과 DANGER의 구분도 실질적으로 약해져 있습니다.

**5. LEAD 이력이 한 프레임 이탈로 삭제됨**

§24와 §31에 정리한 대로, 접지점이 ego lane 경계를 한 프레임만 벗어나도 15샘플 이력과 체류 카운터가 함께 초기화됩니다. LEAD 유지 구간 중앙값이 3프레임인 원인으로 보입니다.

**6. ROI가 고정 좌표**

§3의 두 사다리꼴은 현재 테스트 영상의 카메라 장착 위치에 맞춰 조정한 값입니다. 차선 검출을 수행하지 않으므로 곡선로, 차선 변경, 다른 카메라 위치에서는 실제 주행 차선과 어긋납니다.

**7. 장면 전환 시 Tracker 미초기화**

§29-3에 정리한 대로, 장면 전환이 감지되어도 `MultiObjectTracker`는 초기화되지 않습니다. 이전 장면의 Track ID가 새 장면 객체로 이월됩니다.

**8. 컷인 차량 대응 부재**

§22-2의 횡이동 이력이 계산하는 `outwardDrift`의 음수 방향(차선 중앙으로 파고드는 차량)을 사용하지 않고 있습니다. 컷인 차량은 정의상 방금 차선에 진입한 차량이므로 0.5초 체류 조건을 만족할 수 없고, 결과적으로 가장 위험한 시나리오가 가장 늦게 판정됩니다.

**9. 실시간 처리 불가**

2.39 FPS는 오프라인 후처리 속도입니다. 시간 관련 임계값이 프레임 단위 또는 FPS 환산 기반이므로, 실제 실행 속도가 다른 환경에 그대로 이식하면 경고 시점이 달라집니다.

**10. 장시간 재식별 불가**

Track 보관 시간(20프레임)보다 오래 검출이 끊기면 새 ID가 생성됩니다. 현재 추적기는 위치와 크기 중심이라 이전 차량의 색상이나 외형을 비교할 수 없습니다.

**11. 특정 테스트 영상에 맞춘 Rule-based 조건**

§27의 train 복구 조건, §29의 장면 전환 임계값, §22의 체류·추월 기준은 모두 현재 테스트 영상의 특정 구간에서 역산한 값입니다. 코드 주석에 각 값의 출처 구간을 기록해뒀습니다.

| 값 | 출처 |
| --- | --- |
| ego lane 체류 0.5초 | 35초 길가 트럭 LEAD 오선택 |
| 추월 판정 0.02/frame | 35초 추월 차량 DANGER 오경고 |
| 경고 최소 높이 36px / 증가율 1.08 | 14.98~15.58초 원거리 오경고 |
| CAUTION 배너 8프레임 | 10.57초 주의 상태가 5프레임만 지속 |
| shiftRatio 0.50 | 13초 / 17초 햇빛·노출 변화 오검출 |
| crop conf 0.20 | 38초 ID:60 오검출 |
| train 복구 7개 조건 | 21.9초 우측 탱크로리 |

## 다음 계획

**우선 수정 (60줄 내외)**

- ego lane 이탈 시 3~5프레임 유예를 두어 LEAD 이력 보존 (한계 5)
- 박스 하단 절단 감지 및 종횡비 기반 높이 외삽 (한계 2)
- `MultiObjectTracker::reset()` 추가 및 장면 전환 시 호출 (한계 7)
- `RiskAnalyzer` 내부 조건과 `main.cpp` 필터의 중복 제거 (한계 4)
- `outwardDrift` 음수 방향을 이용한 컷인 차량 체류 조건 면제 (한계 8)

**정량 평가 체계 구축**

- §31의 카운터를 프로그램 내부에 구현하고 종료 시 출력
- 실제 위험 상황이 포함된 영상 확보 (AI Hub 교통사고 영상 데이터 신청 완료)
- 충돌 프레임 기준으로 경고 적시성 지표 정의 (TTC 2.0초 이전 DANGER 발생 여부)
- 위험 없는 구간 100초당 오경고 횟수 정의
- 위 두 지표로 §28의 임계값 재조정

**객체 추적 평가**

- 동일 차량 ID 유지율 측정
- ID Switch 발생 횟수 기록
- Track 생성·삭제·복구 로그 저장
- Re-ID 특징 결합 검토

**위험 분석 개선**

- 차선 검출 기반 동적 ego lane (한계 6)
- 차선 중심과 차량 횡방향 거리 반영
- 차량 합류 상황 감지
- 실제 거리 추정을 위한 카메라 보정
- Optical Flow 기반 접근 방향 보조
- Ego-motion 보정 검토

**객체 검출 개선**

- YOLOv8n과 YOLOv8s 정확도·속도 비교
- 입력 크기 640·960 비교
- 역광·부분 노출 차량 데이터 수집
- 탱크로리·버스·대형 차량 파인튜닝 (한계 11의 train 복구 조건 대체)
- 클래스 오분류 Confusion Matrix 작성

**성능 최적화**

- 원거리 crop 추론 주기 조절 (매 프레임 → N프레임마다)
- 매 프레임 추론과 프레임 스킵 + 추적 보간 방식 비교
- 영상 읽기·추론·저장 멀티스레드 분리
- ONNX Runtime 또는 OpenVINO 비교
- CPU와 GPU/NPU 실행 성능 비교
- 시간 임계값을 전부 초 단위로 정규화 (한계 9)

**문서화**

- README에 전체 파이프라인 다이어그램 추가
- 결과 영상 GIF 또는 짧은 MP4 추가
- Step별 결과 비교 이미지 추가
- Troubleshooting과 Development Log 연결
- 빌드·실행 환경과 의존성 명시