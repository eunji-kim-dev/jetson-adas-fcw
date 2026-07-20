# Development Log

Blackbox ADAS 프로젝트의 기능 구현 과정과 단계별 기술적 의사결정을 기록합니다.

---

## 1. 프로젝트 목표 정의

블랙박스 영상을 입력받아 차량과 보행자를 검출하고, 객체의 이동을 프레임 간 추적한 뒤 주행 경로 내 위험 상황을 판단하는 C++ 기반 영상처리 파이프라인을 구현하는 것을 목표로 설정했습니다.

전체 처리 구조는 다음과 같습니다.

```text
블랙박스 영상 입력
→ 프레임 전처리
→ YOLO 객체 검출
→ 객체 추적 및 ID 부여
→ 주행 ROI 진입 판정
→ 위험 이벤트 판단
→ 결과 영상 및 성능 정보 출력
```

주요 기술 스택은 다음과 같습니다.

```text
C++17
OpenCV
OpenCV DNN
YOLOv8 ONNX
Kalman Filter
CMake
Ubuntu Linux
```

---

## 2. C++ 영상 입출력 파이프라인 구현

OpenCV의 `VideoCapture`와 `VideoWriter`를 사용해 입력 영상을 프레임 단위로 읽고, 처리된 프레임을 새로운 영상으로 저장하는 기본 파이프라인을 구현했습니다.

구현 항목은 다음과 같습니다.

- 입력 영상 열기 및 유효성 검사
- 원본 영상의 해상도, FPS, 전체 프레임 수 확인
- 프레임 단위 반복 처리
- 결과 영상 저장
- 전체 처리 시간 측정
- 초당 처리 프레임 수 계산

처리 성능은 `std::chrono::steady_clock`을 이용해 측정했습니다.

```cpp
const auto processingStart =
    std::chrono::steady_clock::now();

// 프레임 처리

const auto processingEnd =
    std::chrono::steady_clock::now();
```

이를 통해 원본 영상의 재생 FPS와 프로그램의 실제 처리 FPS를 구분해 출력하도록 구성했습니다.

---

## 3. 주행 경로 ROI 구현

화면 전체에 검출된 객체를 모두 위험 객체로 판단하지 않도록, 차량의 예상 진행 경로를 나타내는 사다리꼴 ROI를 구현했습니다.

ROI 좌표는 고정 픽셀값이 아니라 영상의 가로와 세로 비율을 이용해 계산했습니다.

```cpp
const std::vector<cv::Point> drivingRoi = {
    cv::Point(
        static_cast<int>(width * 0.40),
        static_cast<int>(height * 0.65)
    ),
    cv::Point(
        static_cast<int>(width * 0.60),
        static_cast<int>(height * 0.65)
    ),
    cv::Point(
        static_cast<int>(width * 0.90),
        static_cast<int>(height * 0.95)
    ),
    cv::Point(
        static_cast<int>(width * 0.10),
        static_cast<int>(height * 0.95)
    )
};
```

이 방식으로 입력 영상의 해상도가 달라져도 유사한 비율의 ROI가 생성되도록 했습니다.

ROI는 원본 프레임을 완전히 가리지 않도록 반투명 오버레이로 시각화했습니다.

```text
오버레이 프레임 18%
+
원본 프레임 82%
```

초기 ROI는 윗변이 지평선 가까이까지 올라가 있어 실제 위험 판단 영역보다 넓었습니다. 영상의 카메라 시점을 기준으로 윗변 위치를 `height * 0.55`에서 `height * 0.65`로 조정했습니다.

---

## 4. YOLOv8 ONNX 객체 검출 구현

Python 기반 추론 코드에 의존하지 않고 C++ 실행 파일 내부에서 객체 검출을 수행하기 위해 YOLOv8n 모델을 ONNX 형식으로 변환하고 OpenCV DNN에 연결했습니다.

모델 로드는 다음 방식으로 구현했습니다.

```cpp
cv::dnn::Net net =
    cv::dnn::readNetFromONNX(
        "models/yolov8n.onnx"
    );
```

초기 구현에서는 CPU 추론을 기준으로 설정했습니다.

```cpp
net.setPreferableBackend(
    cv::dnn::DNN_BACKEND_OPENCV
);

net.setPreferableTarget(
    cv::dnn::DNN_TARGET_CPU
);
```

ADAS 분석 대상으로 사용할 클래스는 COCO 데이터셋의 80개 클래스 중 다음 항목으로 제한했습니다.

```text
person
bicycle
car
motorcycle
bus
truck
```

---

## 5. Letterbox 전처리 및 좌표 복원

원본 블랙박스 영상은 일반적으로 가로가 긴 비율이지만 YOLO 모델의 입력 크기는 `640 × 640`입니다.

원본 영상을 정사각형으로 직접 resize하면 객체의 가로세로 비율이 변형되므로, 비율을 유지한 채 크기를 조정한 뒤 남는 영역에 여백을 추가하는 Letterbox 전처리를 구현했습니다.

처리 과정은 다음과 같습니다.

```text
원본 영상 비율 확인
→ 비율을 유지한 채 축소
→ 남는 영역을 114 색상값으로 패딩
→ 640 × 640 입력 생성
```

모델 출력 좌표는 다음 순서로 원본 영상 좌표로 복원했습니다.

```text
YOLO 출력 좌표
→ Letterbox 패딩 제거
→ resize 배율로 나누기
→ 원본 영상 범위로 좌표 제한
```

영상 범위를 벗어나는 바운딩 박스는 `std::clamp()`를 이용해 잘라냈습니다.

---

## 6. 클래스별 NMS 구현

YOLO는 하나의 객체 주변에 여러 개의 유사한 바운딩 박스를 출력할 수 있으므로 OpenCV의 `NMSBoxes()`를 사용해 중복 검출을 제거했습니다.

서로 다른 클래스의 박스가 함께 제거되는 것을 방지하기 위해 전체 검출 결과에 한 번 적용하지 않고 클래스별로 분리해 NMS를 수행했습니다.

```text
검출 결과를 classId별로 분리
→ 각 클래스 내부에서 NMS 수행
→ 남은 결과를 하나의 Detection 목록으로 결합
```

이를 통해 차량과 트럭처럼 서로 다른 클래스로 분류된 인접 객체가 하나의 박스로 제거되는 가능성을 줄였습니다.

---

## 7. 객체 검출과 ROI 시각화 순서 개선

초기 구현에서는 ROI 오버레이를 프레임에 먼저 적용한 뒤 YOLO 추론을 수행했습니다.

```text
ROI 오버레이
→ YOLO 추론
```

이 구조에서는 ROI 내부 객체의 픽셀 색상이 초록색으로 변경된 상태로 모델에 입력되어, 정면 차량의 검출 성능이 저하되었습니다.

이를 다음 순서로 변경했습니다.

```text
원본 프레임 YOLO 추론
→ ROI 오버레이
→ 검출 결과 시각화
```

객체 검출용 프레임과 결과 표시용 프레임의 역할을 분리하면서 ROI 내부와 외부 객체를 모두 안정적으로 검출할 수 있게 됐습니다.

자세한 내용은 [Troubleshooting](troubleshooting.md)에 기록했습니다.

---

## 8. 원거리 차량 검출 개선

초기 신뢰도 임계값은 다음과 같았습니다.

```cpp
constexpr float confidenceThreshold = 0.40F;
```

원거리 차량은 바운딩 박스가 작고 시각적 특징이 적어 신뢰도가 상대적으로 낮게 출력됐습니다.

원거리 차량 검출 결과를 유지하기 위해 임계값을 다음과 같이 조정했습니다.

```cpp
constexpr float confidenceThreshold = 0.25F;
```

임계값을 낮추면 원거리 객체 검출률은 높아지지만 오검출 가능성도 증가할 수 있으므로, 향후 다양한 도로 영상에서 precision과 recall을 함께 비교할 예정입니다.

---

## 9. ROI 내부 객체 판정

단순히 바운딩 박스가 ROI와 일부 겹쳤다는 이유만으로 주행 경로 내부 객체로 판단하지 않도록, 바운딩 박스의 하단 중앙점을 객체의 도로 접지점으로 사용했습니다.

```cpp
const cv::Point groundPoint(
    box.x + box.width / 2,
    box.y + box.height
);
```

접지점이 ROI 내부에 있는지는 `pointPolygonTest()`로 판단했습니다.

```cpp
const bool insideRoi =
    cv::pointPolygonTest(
        drivingRoi,
        groundPoint,
        false
    ) >= 0.0;
```

표시 방식은 다음과 같이 구분했습니다.

```text
ROI 외부 객체
→ 주황색 바운딩 박스

ROI 내부 객체
→ 빨간색 바운딩 박스
→ IN ROI 라벨
→ 경고 문구 출력
```

현재 경고는 실제 충돌 가능성을 의미하지 않고, 객체가 주행 ROI에 진입했음을 나타내는 1차 이벤트입니다.

---

## 10. 프레임별 추론 성능 측정

YOLO 추론 시작과 종료 시각을 별도로 측정해 프레임별 추론 지연시간을 계산했습니다.

```cpp
const auto inferenceStart =
    std::chrono::steady_clock::now();

detections = detectObjects(
    net,
    frame,
    confidenceThreshold,
    nmsThreshold
);

const auto inferenceEnd =
    std::chrono::steady_clock::now();
```

측정 결과를 이용해 다음 정보를 영상과 터미널에 출력했습니다.

```text
현재 프레임 번호
프레임별 추론 시간
전체 평균 추론 시간
전체 처리 FPS
ROI 내부 객체 수
```

CPU 추론 환경에서는 프레임별 처리 시간이 대략 `100~300ms` 범위로 측정됐습니다.

향후 입력 해상도, 모델 크기, 프레임 스킵에 따른 성능 차이를 비교할 예정입니다.

---

## 11. Kalman Filter 기반 객체 추적 구현

YOLO는 각 프레임을 독립적으로 검출하므로 동일 차량인지 직접 구분하지 않습니다.

여러 프레임의 검출 결과를 연결하기 위해 `MultiObjectTracker` 클래스를 별도로 구현했습니다.

프로젝트 구조는 다음과 같이 분리했습니다.

```text
include/
├── Detection.hpp
└── MultiObjectTracker.hpp

src/
├── main.cpp
└── MultiObjectTracker.cpp
```

초기 Kalman Filter의 상태값은 다음과 같이 구성했습니다.

```text
중심 x
중심 y
x 방향 속도
y 방향 속도
박스 너비
박스 높이
```

YOLO에서 직접 얻을 수 있는 측정값은 다음 네 가지입니다.

```text
중심 x
중심 y
박스 너비
박스 높이
```

Kalman Filter는 현재까지의 위치와 속도를 이용해 다음 프레임의 객체 위치를 예측하고, 현재 YOLO 검출 박스로 상태를 보정합니다.

---

## 12. IoU 기반 검출-추적 매칭

초기 추적 방식은 Kalman Filter가 예측한 박스와 현재 검출 박스의 IoU를 계산하고, 일정 기준 이상 겹치면 같은 객체로 연결하는 방식으로 구현했습니다.

```text
이전 Track 예측
→ 현재 Detection과 IoU 계산
→ IoU가 높은 조합부터 매칭
→ 연결되지 않은 객체에 새 ID 발급
```

차량이 일시적으로 다음과 같이 다르게 분류되는 경우에도 동일 객체로 연결할 수 있도록 차량 계열 클래스의 일시적인 변경을 허용했습니다.

```text
car
bus
truck
```

---

## 13. Tracking ID 불안정 문제 1차 개선

원거리 차량은 바운딩 박스가 작아 몇 픽셀만 이동해도 IoU가 크게 감소했습니다.

그 결과 동일 차량에도 새로운 ID가 발급되는 문제가 발생했습니다.

IoU 단독 매칭을 보완하기 위해 다음 정보를 추가했습니다.

```text
바운딩 박스 IoU
중심점 사이 거리
객체 크기에 따른 허용 이동 거리
검출 신뢰도
검출 누락 프레임 수
```

두 박스가 충분히 겹치거나 중심점이 허용 거리 이내에 있으면 같은 객체 후보로 판단하도록 수정했습니다.

```text
IoU >= 기준값
또는
중심점 거리 <= 허용 거리
```

1차 개선 후 ID가 유지되는 구간은 증가했으나, 이전에 놓쳤던 ID가 현재 차량과 다시 연결되는 현상이 남았습니다.

---

## 14. Tracking ID 가로채기 문제 분석 및 2차 개선

추적 목록에는 YOLO가 일시적으로 놓친 객체의 ID가 일정 프레임 동안 유지됩니다.

초기 구조에서는 다음 두 종류의 Track이 같은 우선순위로 현재 Detection과 경쟁했습니다.

```text
직전 프레임까지 정상적으로 연결된 Track
검출이 여러 프레임 누락된 과거 Track
```

그 결과 오래된 Track이 현재 차량과 우연히 가까워졌을 때 정상 ID 대신 객체를 가져가는 문제가 발생했습니다.

이를 개선하기 위해 매칭 과정을 두 단계로 분리했습니다.

### 1차 매칭

```text
직전 프레임까지 정상적으로 이어진 Track
→ 현재 Detection에 우선 매칭
```

### 2차 매칭

```text
일시적으로 놓친 Track
→ 1차 매칭 후 남은 Detection에만 재연결
```

놓쳤던 ID에는 다음과 같은 더 엄격한 조건을 적용했습니다.

```text
중심점 거리 제한 강화
박스 면적 변화 제한
재연결 가능한 누락 프레임 수 제한
누락 프레임 수에 따른 매칭 감점
```

또한 추적 범주를 다음처럼 분리했습니다.

```text
사륜 차량
car, bus, truck

이륜 차량
bicycle, motorcycle

보행자
person
```

---

## 15. Kalman Filter 상태 확장

차량이 카메라에 접근하면 바운딩 박스의 중심 위치뿐 아니라 너비와 높이도 지속적으로 증가합니다.

기존 Kalman Filter는 너비와 높이 자체만 상태로 저장하고 크기 변화속도는 예측하지 못했습니다.

이를 보완하기 위해 상태값을 6개에서 8개로 확장했습니다.

```text
중심 x
중심 y
x 방향 속도
y 방향 속도
박스 너비
박스 높이
너비 변화속도
높이 변화속도
```

이를 통해 차량이 접근하거나 멀어질 때 다음 프레임의 바운딩 박스 크기 변화도 함께 예측하도록 개선했습니다.

---

## 현재 구현 상태

현재까지 구현된 파이프라인은 다음과 같습니다.

```text
블랙박스 영상 입력
→ 영상 정보 확인
→ Letterbox 전처리
→ YOLOv8 ONNX 추론
→ 클래스 필터링
→ 클래스별 NMS
→ 원본 영상 좌표 복원
→ Kalman Filter 위치·크기 예측
→ 검출 결과와 Track 매칭
→ 객체별 ID 부여
→ ROI 접지점 판정
→ 상태 및 경고 시각화
→ 결과 영상 저장
→ Latency 및 FPS 출력
```

---

## 다음 개발 계획

### 객체 추적 개선

- 2단계 매칭 방식 결과 검증
- 동일 차량의 ID 유지율 측정
- Hungarian Algorithm 기반 전역 매칭 검토
- 저신뢰도 검출을 재활용하는 ByteTrack 방식 검토
- 카메라 자체 움직임 보정 검토

### 위험 이벤트 판단

- 객체별 바운딩 박스 면적 변화 기록
- 접지점의 수직 이동량 계산
- ROI 내부 체류 프레임 수 계산
- 전방 객체 접근 여부 판정
- Rule-based 전방 충돌 위험 이벤트 구현

### 성능 최적화

- YOLO 입력 크기별 정확도 및 FPS 비교
- 매 프레임 추론과 프레임 스킵 방식 비교
- 영상 읽기와 추론 분리를 위한 멀티스레드 구조 검토
- CPU와 NPU 또는 GPU 실행 성능 비교