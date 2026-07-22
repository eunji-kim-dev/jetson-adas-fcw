# Blackbox ADAS (C++)

블랙박스 영상에서 차량과 보행자를 검출하고, 프레임 간 객체를 추적해 선행 차량의 접근 위험을 분석하는 C++ 기반 ADAS 프로젝트입니다.

YOLOv8 ONNX와 OpenCV DNN으로 객체를 검출하고, Kalman Filter 기반 다중 객체 추적과 TTC-P(TTC Proxy) 계산을 통해 선행 차량의 위험 상태를 시각화합니다.

## 주요 기능

- YOLOv8 ONNX 기반 차량·보행자 검출
- OpenCV DNN을 이용한 C++ 단일 실행 파이프라인
- Kalman Filter 기반 다중 객체 추적
- 고·저신뢰도 Detection을 활용한 ID 유지
- 차량 그룹 단위 NMS
- 주행 ROI와 ego lane 분리
- 선행 차량 LEAD 선택
- 바운딩 박스 변화 기반 TTC-P 계산
- SAFE / CAUTION / DANGER 위험 단계 판단
- 원거리 차량 오경고 필터링
- 장면 전환 감지 및 이전 상태 초기화
- 프레임별 추론 시간과 전체 처리 FPS 출력
- 한국어 상태 정보 및 경고 시각화

## 처리 흐름

```
블랙박스 영상 입력
→ Letterbox 전처리
→ YOLOv8 ONNX 객체 검출
→ 차량 그룹 단위 NMS
→ Kalman Filter 기반 객체 추적
→ 객체별 Tracking ID 부여
→ 주행 ROI 및 ego lane 판정
→ 선행 차량 선택
→ TTC-P 계산
→ 위험 단계 판단
→ 결과 영상 저장
```

## 기술 스택

| 구분 | 기술 |
|---|---|
| Language | C++17 |
| Computer Vision | OpenCV |
| Object Detection | YOLOv8 ONNX |
| Inference | OpenCV DNN |
| Tracking | Kalman Filter, Hungarian Matching |
| Build | CMake |
| Environment | Ubuntu, WSL2 |
| Version Control | Git, GitHub |

## 프로젝트 구조

```
blackbox-adas-cpp/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── Detection.hpp
│   ├── MultiObjectTracker.hpp
│   └── RiskAnalyzer.hpp
├── src/
│   ├── main.cpp
│   ├── MultiObjectTracker.cpp
│   └── RiskAnalyzer.cpp
├── models/
│   └── yolov8n.onnx
├── videos/
│   └── input.mp4
├── results/
└── docs/
    ├── devlog.md
    └── troubleshooting.md
```

## 실행 환경

- Windows 11 + WSL2 (Ubuntu)
- CMake
- OpenCV 4.x
- CPU 기반 YOLO 추론

## 설치 및 실행

### 의존성 설치

```bash
sudo apt update

sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libopencv-contrib-dev \
    fonts-nanum
```

### 모델 및 입력 영상 준비

YOLOv8n ONNX 모델은 다음 경로에 둡니다.

```
models/yolov8n.onnx
```

기본 입력 영상 경로는 다음과 같습니다.

```
videos/input.mp4
```

### 빌드

```bash
cmake -S . -B build
cmake --build build -j
```

CMake 설정이 바뀐 경우에는 기존 빌드 폴더를 지우고 다시 빌드합니다.

```bash
rm -rf build

cmake -S . -B build
cmake --build build -j
```

### 실행

기본 입력 영상으로 실행합니다.

```bash
./build/adas
```

다른 영상을 쓰려면 경로를 인자로 넘깁니다.

```bash
./build/adas videos/another_video.mp4
```

## 결과

실행이 끝나면 `results/` 폴더에 분석 결과 영상이 생성됩니다. 최종 영상에는 다음 정보가 표시됩니다.

- 객체 클래스
- Tracking ID
- 선행 차량 LEAD 표시
- TTC-P
- 위험 상태
- 프레임별 추론 시간
- 현재 객체 수
- 한국어 경고 메시지

## 주요 구현 내용

**객체 검출**

YOLOv8n ONNX 모델을 OpenCV DNN으로 불러와 CPU에서 추론합니다. 사용하는 클래스는 person, bicycle, car, motorcycle, bus, truck입니다.

**다중 객체 추적**

Kalman Filter로 다음 프레임의 객체 위치와 크기를 예측합니다. 매칭에는 IoU, 중심점 거리, 박스 크기 변화, 클래스 그룹, 검출 신뢰도, 누락 프레임 수를 함께 사용합니다.

**선행 차량 선택**

넓은 주행 ROI와 실제 위험 분석용 ego lane을 분리하고, ego lane 안에서 화면 중앙과 가까운 차량을 선행 차량으로 선택합니다.

**TTC-P**

단안 블랙박스 영상만으로는 실제 거리를 계산하기 어렵기 때문에, 선행 차량 바운딩 박스 높이의 증가율을 이용한 TTC-P를 사용합니다. TTC-P는 실제 거리 기반 TTC가 아니라 영상 스케일 변화 기반 접근 위험 지표입니다.

## 문제 해결 사례

- ROI 오버레이가 YOLO 입력을 오염시키는 문제 해결
- 원거리 차량 검출 누락 개선
- 동일 차량의 Tracking ID 반복 변경 개선
- 오래된 Track이 현재 객체의 ID를 가로채는 문제 개선
- car, bus, truck 중복 검출 감소
- 탱크로리의 train 오분류 원인 진단 및 제한적 복구
- 원거리 차량 박스 흔들림으로 인한 오경고 감소
- 장면 전환 뒤 이전 경고가 남는 문제 해결

상세한 원인 분석과 수정 과정은 아래 문서에 정리했습니다.

- [Development Log](docs/devlog.md)
- [Troubleshooting](docs/troubleshooting.md)

## 한계 및 개선 방향

현재 TTC-P는 바운딩 박스 크기 변화를 이용한 상대적 위험 지표입니다. 앞으로는 다음 항목을 개선할 계획입니다.

- 카메라 보정 기반 실제 거리 추정
- Ego-motion 보정
- Re-ID 기반 장시간 객체 재식별
- YOLOv8s 이상 모델 성능 비교
- 부분 노출 차량 데이터 파인튜닝
- GPU 또는 NPU 기반 추론 최적화
- 다양한 주행 영상 기반 정량 평가
