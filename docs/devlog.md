# Development Log

Blackbox ADAS 프로젝트를 구현하면서 진행한 기능 개발과 그 과정에서의 기술적 판단들을 기록합니다.

---

## 1. 프로젝트 목표

블랙박스 영상을 입력받아 차량과 보행자를 검출하고, 프레임 간 객체 이동을 추적해 주행 경로 안의 위험 상황을 판단하는 C++ 영상처리 파이프라인을 만드는 것이 목표였습니다.

전체 흐름은 다음과 같이 잡았습니다.

```text
블랙박스 영상 입력
→ 프레임 전처리
→ YOLO 객체 검출
→ 객체 추적 및 ID 부여
→ 주행 ROI 진입 판정
→ 위험 이벤트 판단
→ 결과 영상 및 성능 정보 출력
```

사용한 기술 스택:

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

## 2. 영상 입출력 파이프라인

`VideoCapture`와 `VideoWriter`로 프레임 단위 입출력부터 잡았습니다. 입력 영상 유효성 검사, 해상도·FPS·전체 프레임 수 확인, 프레임 반복 처리, 결과 저장까지가 기본 골격입니다.

처리 시간은 `std::chrono::steady_clock`으로 측정했습니다.

```cpp
const auto processingStart =
    std::chrono::steady_clock::now();

// 프레임 처리

const auto processingEnd =
    std::chrono::steady_clock::now();
```

원본 영상의 재생 FPS와 실제 처리 FPS는 다른 값이라, 둘을 구분해서 따로 출력하도록 했습니다.

---

## 3. 주행 경로 ROI

화면에 잡히는 객체를 전부 위험 객체로 취급하면 의미가 없어서, 차량이 실제로 지나갈 경로를 나타내는 사다리꼴 ROI를 정의했습니다.

좌표는 고정 픽셀이 아니라 영상 가로세로 비율로 계산합니다.

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

이렇게 하면 입력 해상도가 바뀌어도 비슷한 비율의 ROI가 나옵니다.

시각화는 원본 프레임을 완전히 가리지 않도록 반투명 오버레이로 처리했습니다.

```text
오버레이 프레임 18%
+
원본 프레임 82%
```

처음엔 ROI 윗변이 지평선 가까이까지 올라가 있어서 실제로 위험을 판단할 영역보다 훨씬 넓었습니다. `height * 0.55`였던 값을 `height * 0.65`로 낮춰서 카메라 시점에 맞게 좁혔습니다.

---

## 4. YOLOv8 ONNX 연동

Python 추론 스크립트에 의존하지 않고 C++ 실행 파일 안에서 검출을 끝내고 싶어서, YOLOv8n을 ONNX로 변환한 뒤 OpenCV DNN에 붙였습니다.

```cpp
cv::dnn::Net net =
    cv::dnn::readNetFromONNX(
        "models/yolov8n.onnx"
    );
```

일단은 CPU 추론으로 시작했습니다.

```cpp
net.setPreferableBackend(
    cv::dnn::DNN_BACKEND_OPENCV
);

net.setPreferableTarget(
    cv::dnn::DNN_TARGET_CPU
);
```

COCO 80개 클래스 중 ADAS와 관련 있는 것만 남겼습니다.

```text
person
bicycle
car
motorcycle
bus
truck
```

---

## 5. Letterbox 전처리와 좌표 복원

블랙박스 영상은 보통 가로가 긴 비율인데, YOLO 입력은 `640 × 640` 정사각형입니다. 그냥 resize하면 객체 비율이 뒤틀리기 때문에, 비율은 유지한 채 축소하고 남는 공간에 여백을 채우는 Letterbox 방식을 썼습니다.

```text
원본 영상 비율 확인
→ 비율을 유지한 채 축소
→ 남는 영역을 114 색상값으로 패딩
→ 640 × 640 입력 생성
```

모델 출력 좌표는 반대로 되돌려야 합니다.

```text
YOLO 출력 좌표
→ Letterbox 패딩 제거
→ resize 배율로 나누기
→ 원본 영상 범위로 좌표 제한
```

영상 밖으로 나가는 박스는 `std::clamp()`로 잘랐습니다.

---

## 6. 클래스별 NMS

YOLO는 한 객체 주변에 겹치는 박스를 여러 개 뱉는 경우가 많아서 `NMSBoxes()`로 중복을 제거했습니다.

전체 검출 결과에 한 번에 NMS를 걸면 서로 다른 클래스의 박스끼리도 지워질 수 있어서, classId별로 나눈 다음 각각 NMS를 돌리고 다시 합치는 방식으로 바꿨습니다.

```text
검출 결과를 classId별로 분리
→ 각 클래스 내부에서 NMS 수행
→ 남은 결과를 하나의 Detection 목록으로 결합
```

차량과 트럭처럼 인접한 다른 클래스 객체가 한쪽으로 묻혀 사라지는 걸 줄이려는 목적입니다.

---

## 7. 검출과 ROI 시각화 순서 조정

처음엔 ROI 오버레이를 프레임에 먼저 씌우고 그 위에 YOLO 추론을 돌렸습니다.

```text
ROI 오버레이
→ YOLO 추론
```

이러면 ROI 안에 있는 객체는 이미 초록색으로 픽셀이 바뀐 상태로 모델에 들어가서, 정면 차량 검출 성능이 떨어졌습니다.

순서를 바꿨습니다.

```text
원본 프레임 YOLO 추론
→ ROI 오버레이
→ 검출 결과 시각화
```

검출용 프레임과 화면 표시용 프레임의 역할을 나누고 나니 ROI 안팎 객체가 모두 안정적으로 잡혔습니다. 자세한 내용은 [Troubleshooting](troubleshooting.md)에 정리해뒀습니다.

---

## 8. 원거리 차량 검출 보완

처음 신뢰도 임계값은 이거였습니다.

```cpp
constexpr float confidenceThreshold = 0.40F;
```

원거리 차량은 박스가 작고 특징도 적어서 신뢰도가 낮게 나오다 보니, 이 기준에서 자꾸 걸러졌습니다.

```cpp
constexpr float confidenceThreshold = 0.25F;
```

0.25로 낮췄더니 원거리 검출은 늘었는데, 오검출도 같이 늘어날 여지가 있어서 나중에 다양한 도로 영상으로 precision·recall을 같이 비교해볼 계획입니다.

---

## 9. ROI 내부 판정 방식

바운딩 박스가 ROI에 조금이라도 걸치면 내부로 판단하는 건 너무 느슨해서, 박스 하단 중앙점을 객체의 도로 접지점으로 잡아 그 점만으로 판정하도록 바꿨습니다.

```cpp
const cv::Point groundPoint(
    box.x + box.width / 2,
    box.y + box.height
);
```

접지점이 ROI 안에 있는지는 `pointPolygonTest()`로 확인합니다.

```cpp
const bool insideRoi =
    cv::pointPolygonTest(
        drivingRoi,
        groundPoint,
        false
    ) >= 0.0;
```

표시는 이렇게 나눴습니다.

```text
ROI 외부 객체
→ 주황색 바운딩 박스

ROI 내부 객체
→ 빨간색 바운딩 박스
→ IN ROI 라벨
→ 경고 문구 출력
```

지금 단계의 경고는 실제 충돌 위험을 뜻하는 게 아니라, 객체가 주행 ROI에 들어왔다는 걸 알리는 1차 이벤트 정도입니다.

---

## 10. 프레임별 추론 시간 측정

YOLO 추론 시작·종료 시각을 따로 재서 프레임별 지연시간을 계산했습니다.

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

이 값으로 다음 정보를 화면과 터미널에 함께 출력합니다.

```text
현재 프레임 번호
프레임별 추론 시간
전체 평균 추론 시간
전체 처리 FPS
ROI 내부 객체 수
```

CPU 추론 기준으로는 프레임당 대략 `100~300ms` 정도가 나왔습니다. 이후 입력 해상도, 모델 크기, 프레임 스킵에 따라 성능이 어떻게 바뀌는지 비교해볼 예정입니다.

---

## 11. Kalman Filter 기반 추적 구현

YOLO는 프레임마다 독립적으로 검출만 할 뿐 동일 차량인지는 구분하지 않기 때문에, 여러 프레임의 검출 결과를 이어붙이는 `MultiObjectTracker` 클래스를 따로 만들었습니다.

```text
include/
├── Detection.hpp
└── MultiObjectTracker.hpp

src/
├── main.cpp
└── MultiObjectTracker.cpp
```

초기 Kalman Filter 상태값은 다음 6개였습니다.

```text
중심 x
중심 y
x 방향 속도
y 방향 속도
박스 너비
박스 높이
```

YOLO에서 바로 얻는 측정값은 이 4개뿐입니다.

```text
중심 x
중심 y
박스 너비
박스 높이
```

Kalman Filter가 지금까지의 위치·속도로 다음 프레임 위치를 예측하고, 실제 YOLO 검출값으로 그 예측을 보정하는 구조입니다.

---

## 12. IoU 기반 매칭

Kalman Filter가 예측한 박스와 현재 프레임 검출 박스의 IoU를 계산해서, 일정 기준 이상 겹치면 같은 객체로 이어 붙이는 방식으로 시작했습니다.

```text
이전 Track 예측
→ 현재 Detection과 IoU 계산
→ IoU가 높은 조합부터 매칭
→ 연결되지 않은 객체에 새 ID 발급
```

차량이 프레임마다 car/bus/truck으로 다르게 분류되는 경우가 있어서, 이 세 클래스 사이의 일시적인 분류 변경은 같은 객체로 봐주도록 허용했습니다.

---

## 13. Tracking ID 불안정 문제 1차 개선

원거리 차량은 박스가 작다 보니 몇 픽셀만 움직여도 IoU가 크게 떨어졌고, 그러다 보니 같은 차량인데도 새 ID가 자꾸 발급됐습니다.

IoU만으로는 부족해서 다음 정보를 같이 봤습니다.

```text
바운딩 박스 IoU
중심점 사이 거리
객체 크기에 따른 허용 이동 거리
검출 신뢰도
검출 누락 프레임 수
```

박스가 충분히 겹치거나, 아니면 중심점 거리가 허용 범위 안이면 같은 객체 후보로 판단하도록 했습니다.

```text
IoU >= 기준값
또는
중심점 거리 <= 허용 거리
```

이 개선으로 ID가 유지되는 구간은 늘었지만, 예전에 놓쳤던 ID가 지금 차량과 다시 이어지는 문제는 남았습니다.

---

## 14. Tracking ID 가로채기 문제 개선

추적 목록에는 YOLO가 잠깐 놓친 객체의 ID도 몇 프레임 동안 그대로 남아 있습니다. 문제는 이게 정상적으로 이어지고 있는 Track과 똑같은 우선순위로 현재 검출 결과를 두고 경쟁한다는 점이었습니다. 그러다 오래된 Track이 우연히 지금 차량 근처로 예측값이 흘러가면, 정상 ID 대신 그 오래된 ID가 객체를 채가는 상황이 생겼습니다.

매칭을 두 단계로 나눴습니다.

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

놓쳤던 ID 쪽에는 더 까다로운 조건을 걸었습니다.

```text
중심점 거리 제한 강화
박스 면적 변화 제한
재연결 가능한 누락 프레임 수 제한
누락 프레임 수에 따른 매칭 감점
```

추적 범주도 나눴습니다.

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

차량이 카메라 쪽으로 다가오면 중심 위치뿐 아니라 박스 너비·높이도 계속 커집니다. 그런데 기존 Kalman Filter는 너비·높이 값 자체만 상태로 갖고 있었고, 그게 얼마나 빠르게 커지는지는 예측하지 못했습니다.

상태값을 6개에서 8개로 늘렸습니다.

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

이제 차량이 가까워지거나 멀어질 때 다음 프레임의 박스 크기 변화까지 예측 범위에 들어갑니다.

---

## 현재 구현 상태

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

## 다음 계획

### 객체 추적

- 2단계 매칭 결과 검증
- 동일 차량 ID 유지율 측정
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

- YOLO 입력 크기별 정확도·FPS 비교
- 매 프레임 추론과 프레임 스킵 방식 비교
- 영상 읽기와 추론 분리를 위한 멀티스레드 구조 검토
- CPU와 NPU 또는 GPU 실행 성능 비교