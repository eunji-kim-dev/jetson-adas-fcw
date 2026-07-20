# Blackbox ADAS C++ Project

C++와 OpenCV를 사용하여 블랙박스 영상 기반 ADAS 기능을 구현하는 프로젝트입니다.

## 현재 구현 내용

- Ubuntu WSL 기반 C++ 개발 환경 구성
- CMake 기반 C++17 빌드 시스템 구성
- OpenCV 라이브러리 연동
- OpenCV 이미지 생성 및 파일 저장

## 기술 스택

- C++17
- OpenCV
- CMake
- Ubuntu / WSL

## 프로젝트 구조

- `src/main.cpp`: C++ 소스코드
- `videos/`: 입력 영상 보관
- `results/opencv_test.png`: OpenCV 테스트 결과
- `CMakeLists.txt`: CMake 빌드 설정
- `README.md`: 프로젝트 설명

## 빌드 및 실행

1. `cmake -S . -B build`
2. `cmake --build build`
3. `./build/adas`

## 실행 결과

프로그램을 실행하면 `results/opencv_test.png` 파일이 생성됩니다.

![OpenCV test result](results/opencv_test.png)

## 향후 구현 계획

- 블랙박스 영상 입력 및 출력
- ROI 기반 영상 분석
- 움직임 감지
- 객체 검출
- 객체 추적
- ADAS 이벤트 판단 로직
- FPS 및 Latency 측정

## 주요 문제 해결

- ROI 시각화를 YOLO 추론 전에 적용해 정면 차량 검출률이 저하되는 문제를 발견하고, 원본 프레임 추론 후 ROI를 표시하도록 처리 순서를 수정했습니다.
- 원거리 소형 차량의 검출 누락을 줄이기 위해 confidence threshold를 0.40에서 0.25로 조정했습니다.
- IoU 단독 매칭으로 동일 차량의 Tracking ID가 변경되는 문제를 분석하고, Kalman Filter 예측과 중심점 거리·박스 크기 조건을 결합한 추적 방식으로 개선했습니다.
- 정상 추적 ID와 오래된 ID의 매칭 순서를 분리해 기존 ID가 현재 객체를 가로채는 문제를 줄였습니다.
- 프레임별 추론 시간과 전체 처리 FPS를 측정해 실시간 처리 성능을 수치화했습니다.

상세한 오류 분석과 해결 과정은 다음 문서에 기록했습니다.

- [Troubleshooting](docs/troubleshooting.md)
- [Development Log](docs/devlog.md)