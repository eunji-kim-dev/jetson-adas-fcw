# 개발 작업일지


## 2026-08-24

### 1. ONNX opset 12 확인
- input: `images [1, 3, 640, 640]`
- output: `output0 [1, 84, 8400]`
- static shape 확인
- NMS는 ONNX 내부가 아니라 C++ `NMSBoxes()`에서 처리됨

### 알게 된 점
- 현재 ONNX 자체가 640×640 고정 입력으로 export되어 있음
- 960 입력을 비교하려면 960용 ONNX를 별도로 export해야 함


### 2. 탱크로리 train→truck 보정 로직 제거
- 특정 탱크로리를 train이 아니라 truck으로 처리하도록 넣어둔 예외 로직 제거함
- 해당 로직과 함께 사용하던 불필요한 변수와 점수 처리 코드 정리함
- 코드 길이 약 40줄 감소함
- 컴파일 문법 검사와 경고 검사 통과함
- 실제 영상 실행 결과는 아직 확인 전임

### 알게 된 점
- 기존 로직은 특정 영상의 탱크로리 한 대를 잡기 위해 좌표 조건까지 넣은 임시 보정이었음
- 이런 예외 처리가 남아 있으면 실제 YOLO의 오분류 결과를 제대로 확인하기 어려움
- 성능 실험 전에 제거하는 것이 맞다고 판단함
- 제거 후 해당 탱크로리가 다시 train으로 분류되어 추적되지 않을 수 있음


### 3. 프레임별 CSV 로그 추가 및 기준 결과 저장
- 영상마다 결과 파일이 따로 저장되도록 출력 파일명을 입력 영상 이름 기준으로 변경함
  - 영상 결과: `results/<영상이름>_output.avi`
  - CSV 결과: `results/<영상이름>_frames.csv`
- 각 프레임마다 다음 정보를 CSV에 기록하도록 추가함
    - 프레임 번호 (`frame`)
    - 장면 전환 여부 (`sceneChanged`)
    - YOLO가 검출한 객체 개수 (`numDetections`)
    - YOLO가 검출한 객체 정보 (`detections`)
    - 트래커가 추적 중인 객체 개수 (`numTracks`)
    - 트래커가 추적 중인 객체 정보 (`tracks`)
    - 현재 선행 차량 ID (`activeLeadId`)
    - 위험 단계 (`riskLevel`)
    - TTC (`ttc`)

    #### CSV 로그 읽는 법
    CSV 한 줄은 영상의 한 프레임 상태를 의미

    ##### detections
    ```text
    2 : 400 : 250 : 80 : 60 : 0.9123
    │    │     │     │    │      │
    종류  x     y    너비  높이   신뢰도
    ```
    - 종류(classId): YOLO가 판단한 객체 종류
    - x, y: 객체를 둘러싼 박스의 왼쪽 위 위치
        - 너비, 높이: 객체 박스의 크기
        - 신뢰도: YOLO가 해당 객체라고 판단한 확신 정도
        - `0.9123` → 약 91.23%의 신뢰도

    - 현재 사용하는 COCO 기준 주요 classId
        - `2` → car
        - `5` → bus
        - `6` → train
        - `7` → truck

    객체가 여러 개 검출되면 `;`로 구분

    ##### tracks
    ```text
    51 : 2 : 400 : 250 : 80 : 60
    │    │    │     │     │    │
    ID  종류   x     y    너비  높이
    ```
    - ID: 트래커가 객체에 붙인 고유 번호임
    - 종류: 객체의 classId임
    - x, y: 객체 박스의 왼쪽 위 위치임
    - 너비, 높이: 객체 박스의 크기임

    객체가 여러 개 검출되면 `;`로 구분

    Detection과 Track의 차이:
    Detection = YOLO가 현재 프레임만 보고 새로 발견한 객체임
    Track = 여러 프레임을 이어서 보면서 같은 객체라고 계속 추적하고 있는 결과임
- 같은 영상을 2번 실행한 결과 CSV 내용이 완전히 동일하게 나옴
- 따라서 현재 결과를 기준값으로 저장해두고, 이후 코드 구조를 바꾼 뒤 결과가 달라졌는지 비교할 수 있게 됨

### 알게 된 점
- 탱크로리는 YOLO가 중간중간 `car`, `bus`, `truck`으로 다르게 인식했지만 같은 ID로 계속 추적됨
- 즉 YOLO가 차종을 잠깐 다르게 판단해도 트래커가 바로 다른 차량으로 취급하지는 않음
- 현재 영상에서는 `CAUTION`이 한 번도 나오지 않았고 `DANGER`는 2프레임에서만 나옴
- 따라서 나중에 다른 영상도 사용해서 위험 단계가 제대로 나뉘는지 확인할 필요가 있음
- 전체 1252프레임 중 650프레임에서는 선행 차량이 선택되지 않았음
- 나중에 내 차선 영역(ROI)을 조정할 때 선행 차량을 잡는 범위도 같이 확인할 필요가 있음


### 4. perception/adas/apps 디렉토리 구조로 파일 이동
- 기존 코드를 역할에 따라 세 영역으로 나눠서 파일을 이동함
  - `perception/` → Detection, MultiObjectTracker
  - `adas/` → RiskAnalyzer
  - `apps/fcw_demo/` → main.cpp
- 헤더 include 경로를 소속이 드러나도록 변경함
  - `#include "perception/..."`
  - `#include "adas/..."`
- CMakeLists의 소스 경로와 include 경로를 새 구조에 맞게 수정함
- 아직 CMake 타깃은 기존 `adas` 하나로 유지함
- 프로그램 로직은 변경하지 않음
- 같은 영상을 다시 실행한 결과 baseline MD5가 이전과 동일함
  - `a0006c4a16dbe3f69c178fbc5c1b6b8e`
- 따라서 파일 위치와 include 경로만 바뀌었고 실행 결과는 변하지 않았음을 확인함


### 다음 작업
- CMake 타깃을 `perception_core`, `adas_fcw`, `fcw_demo`로 분리함
  - 의존 방향: `perception_core ← adas_fcw ← fcw_demo`
- `perception_core`에서 `adas` 헤더를 include했을 때 빌드가 실패하는지 확인함
  - 확인 후 테스트용 include는 바로 되돌림
- `perception_demo`를 추가해서 `adas_fcw` 없이도 Detection/Tracking만 링크·실행되는지 확인함
- 같은 기준 영상을 다시 실행해서 baseline MD5가 `a0006c4a16dbe3f69c178fbc5c1b6b8e`와 동일한지 확인함
- 이상 없으면 타깃 분리 작업을 커밋함

## 2026-08-25

### 1. CMake 타깃 분리 및 perception_demo 추가
- 최상위 CMakeLists.txt는 전체 설정과 perception / adas / apps 연결만 담당하도록 단순화
- 각 폴더에 별도의 CMakeLists.txt 추가
- 코드를 다음과 같이 빌드 단위로 분리
  - perception_core: 객체 인식·추적 기능
  - adas_fcw: FCW 위험 판단 기능
  - adas: 실제 ADAS 실행 프로그램
  - perception_demo: perception만 따로 실행해보는 테스트 프로그램
- 의존 관계를 adas → adas_fcw → perception_core 순서로 고정
- perception_core에는 필요한 OpenCV 모듈만 연결하고, DNN·한글 출력·영상 입출력 등은 필요한 앱 쪽에서 연결
- perception 코드에서 일부러 adas/RiskAnalyzer.hpp를 include해 빌드 테스트
  - 헤더를 찾지 못해 컴파일 실패하는 것 확인
  - 즉 perception → adas 방향으로는 접근할 수 없음을 확인
  - 테스트 후 해당 include는 삭제하고 정상 재빌드
- perception_demo를 ADAS 없이 실행해 전체 1252프레임 처리 성공
- 기존 ADAS 결과 CSV의 MD5도 이전과 동일한 것 확인
  - a0006c4a16dbe3f69c178fbc5c1b6b8e

### 이 작업을 한 이유
- perception과 adas를 폴더만 나누는 게 아니라, 실제 빌드 단계에서도 서로 역할을 분리하기 위해
- perception은 객체를 인식하고 추적하는 기능만 담당하고, adas는 그 결과를 받아 FCW 위험 판단만 하도록 만들기 위해
- 나중에 ROS2나 다른 로봇 프로젝트에서도 perception_core를 ADAS와 상관없이 재사용할 수 있게 하기 위해
- 잘못된 방향인 perception → adas 의존성이 생기지 않도록 CMake가 직접 막게 하기 위해
- 9월 Jetson으로 옮길 때 전체 구조를 다시 뜯어고치지 않고, 필요한 앱 쪽 설정만 수정할 수 있게 하기 위해
- 리팩터링 과정에서 기존 ADAS 결과가 바뀌지 않았는지 MD5로 확인하기 위해

### 2. 검출 기능을 perception으로 이동 (YoloDetector 분리)
- 기존 `main.cpp` 안에 있던 객체 검출 코드를 `perception`으로 이동
  - 이미지 크기 맞추기(letterbox)
  - YOLO 객체 검출
  - 멀리 있는 차량을 보기 위한 crop 추가 검출
  - 겹치는 박스 정리(NMS)
  - 중복 박스 제거
  - 너무 작은 객체 제거
- 위 검출 과정을 `YoloDetector`라는 하나의 클래스로 묶음
- YOLO 모델을 실행하는 `cv::dnn::Net`도 `YoloDetector` 내부에서 관리하도록 변경
- 차량/사람 등의 클래스 종류를 판단하는 기능은 `Classes.hpp`로 따로 분리
- 기존 성능 측정 로그 `[PERF]`가 바뀌지 않도록 각 검출 단계의 시간을 `DetectionTiming`으로 반환
- `perception_demo`에도 실제 YOLO 검출을 연결
  - 이제 `ADAS` 없이도 `Detection → Tracking`까지 실제로 실행 가능
- 기존 ADAS 결과와 MD5가 같은지 확인
  - `a0006c4a16dbe3f69c178fbc5c1b6b8e`

### 이 작업을 한 이유
- 객체 검출 코드가 `main.cpp` 안에 있으면 다른 프로그램에서 검출 기능만 따로 사용할 수 없기 때문
- `Detection + Tracking`을 ADAS와 상관없이 사용할 수 있어야 `perception`이 독립된 모듈이 됨
- YOLO 실행 부분을 `YoloDetector` 안에 넣어두면 나중에 OpenCV DNN을 TensorRT로 바꿀 때 **검출기 내부만 수정하고 사용하는 쪽 코드는 그대로 둘 수 있음**
- 차량인지 사람인지 판단하는 공통 기능을 `Classes.hpp`로 빼두면 이후 `adas`에서도 `main.cpp`에 의존하지 않고 사용할 수 있음

### 알게 된 점
- 커밋하기 전에는 반드시 빌드와 baseline 검증을 먼저 해야 함
- 여러 변경사항을 한 번에 섞어서 커밋하기보다 **하나의 작업이 정상 동작하는 상태에서 커밋하는 것이 좋음**
- 이렇게 해야 나중에 결과가 잘못됐을 때 **어느 변경부터 문제가 생겼는지 추적하기 쉬움**