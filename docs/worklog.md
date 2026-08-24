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