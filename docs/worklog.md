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
- `perception`과 `adas`를 폴더뿐 아니라 실제 빌드 단위로 분리하기 위해
- `perception → adas` 같은 잘못된 의존성이 생기지 않도록 CMake에서 구조적으로 막기 위해
- 이후 Jetson이나 ROS2로 옮길 때 필요한 모듈만 독립적으로 사용할 수 있게 하기 위해


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
- `main.cpp`에 있던 객체 검출 기능을 `perception`의 독립 기능으로 만들기 위해
- ADAS 없이도 `Detection → Tracking`을 사용할 수 있게 하기 위해
- 이후 OpenCV DNN을 TensorRT로 바꿀 때 `YoloDetector` 내부만 수정할 수 있게 하기 위해

### 알게 된 점
- 커밋하기 전에는 반드시 빌드와 baseline 검증을 먼저 해야 함
- 여러 변경사항을 한 번에 섞어서 커밋하기보다 **하나의 작업이 정상 동작하는 상태에서 커밋하는 것이 좋음**
- 이렇게 해야 나중에 결과가 잘못됐을 때 **어느 변경부터 문제가 생겼는지 추적하기 쉬움**


### 3. LEAD 선정 로직을 `adas`로 이동 (`LeadSelector` 분리)
- 기존 `main.cpp`에 있던 선행 차량(LEAD) 선정 코드를 `adas/`로 이동
- LEAD를 고르는 과정은 `LeadSelector` 클래스로 따로 분리
  - 차량이 내 차선에 있는지 확인
  - 잠깐 차선을 벗어나도 기존 LEAD를 바로 바꾸지 않도록 처리
  - 차량의 좌우 이동 기록
  - 끼어드는 차량이나 추월 차량 판단
  - 여러 차량 중 어떤 차량을 LEAD로 할지 점수 계산
  - 기존 LEAD보다 확실히 좋은 후보가 있을 때만 교체
  - 화면에서 사라진 차량의 기록 정리
- 장면이 바뀌면 LEAD 관련 상태를 `leadSelector.reset()`으로 한 번에 초기화
- `main.cpp`에서는 LEAD 선정 과정을 직접 하지 않고, `LeadSelector`가 고른 결과만 사용
- 기존 렌더링, 경고 표시, CSV 저장 코드는 그대로 유지
- baseline MD5 일치 확인
  - `a0006c4a16dbe3f69c178fbc5c1b6b8e`

### 이 작업을 한 이유
- LEAD 선정은 일반적인 객체 추적이 아니라 FCW를 위한 판단 로직이므로 `adas`가 담당하도록 하기 위해
- `main.cpp`에 남아 있던 FCW 전용 상태와 판단 코드를 제거하고, `main.cpp`는 각 모듈을 연결하는 역할만 하도록 만들기 위해
- `perception`에는 LEAD 여부나 위험도 같은 ADAS 전용 정보를 넣지 않고 일반적인 검출·추적 정보만 유지하기 위해


### 4. 경고 판단 로직을 `adas`로 이동 (`WarningPolicy` 분리)
- `main.cpp`에 마지막으로 남아 있던 FCW 경고 관련 코드를 `adas/`로 이동
- 경고를 화면에 띄우는 조건을 `WarningPolicy` 클래스로 분리
  - CAUTION은 8프레임 연속 확인 후 표시
  - DANGER는 3프레임 연속 확인 후 표시
  - 한 번 표시된 경고는 0.25초 동안 유지
  - LEAD 차량이 바뀌면 이전 차량의 경고 카운트를 이어받지 않음
- 차량 위치와 추월 여부를 보고 경고를 제한하는 조건도 `WarningPolicy`로 이동
- 장면이 바뀔 때 여러 경고 상태를 각각 초기화하던 코드를 `warningPolicy.reset()`으로 정리
- `main.cpp`에서는 경고 조건을 직접 계산하지 않고 `WarningPolicy`의 결과만 사용하도록 변경
- baseline MD5 일치 확인
  - `a0006c4a16dbe3f69c178fbc5c1b6b8e`
- 분리 작업 후 `main.cpp`가 약 1,130줄에서 606줄로 감소
- FCW 판단 흐름을 다음과 같이 정리
  - `LeadSelector` → 선행 차량 선택
  - `RiskAnalyzer` → 충돌 위험 판단
  - `WarningPolicy` → 실제 경고 표시 여부 결정

### 이 작업을 한 이유
- 위험도 계산과 실제 경고 표시 조건을 서로 다른 역할로 분리하기 위해
- `RiskAnalyzer`는 현재 차량이 얼마나 위험한지를 판단하고, `WarningPolicy`는 그 결과를 언제 실제 경고로 표시할지 담당하도록 하기 위해
- 마지막으로 남아 있던 FCW 판단 코드를 `adas`로 이동해, `main.cpp`에는 영상 입력·화면 출력·각 모듈 연결 같은 실행 코드만 남기기 위해

### 알게 된 점
- 리팩터링은 기존 동작을 유지한 채 내부 구조를 개선하는 작업이라는 것은 알고 있었지만, 실제로 어떤 순서로 진행해야 하는지는 이번 작업을 통해 처음 익힘
- 리팩터링 전에는 기존 결과를 baseline으로 남겨두고, 구조를 바꾼 뒤 같은 결과가 나오는지 비교
- 한 번에 전체를 바꾸는 것이 아니라, 모듈을 단계별로 분리하고 매 단계마다 빌드와 실행으로 확인하면서 진행
- 이번 프로젝트에서는 CSV의 MD5를 비교해 리팩터링 전후 결과가 동일한지 확인
- 정상 동작만 보는 것이 아니라, 일부러 잘못된 의존성을 넣어 빌드가 실패하는지 확인하는 파괴 테스트로 구조가 실제로 강제되는지도 검증
- 리팩터링을 실제로 진행하는 흐름
  `baseline 확보 → 구조 변경 → 빌드 → 실행 → 기존 결과와 비교 → 구조 제약 검증 → 원상복구 및 재빌드 → 커밋`

### 다음 작업
- YOLO 실행 부분을 `InferenceBackend` 인터페이스로 분리
- 현재 OpenCV DNN 방식은 `OpenCVDNNBackend`로 구현
- 실행할 때 `--backend` 옵션으로 추론 방식을 선택할 수 있도록 변경
- 영상 입력 부분을 `FrameSource` 인터페이스로 분리
- 현재 영상 파일 입력은 `VideoFileSource`로 구현
  - `image`
  - `frameSeq`
  - `captureTimestampNs`
  - `captureTimestampSource`
  정보를 프레임과 함께 관리하도록 구성


## 2026-08-26

### 1. YOLO 추론 구조 분리
- `YoloDetector` 안에 직접 들어 있던 OpenCV DNN 추론 코드를 분리
- 공통 인터페이스인 `InferenceBackend`를 만들고, 현재 방식은 `OpenCVDNNBackend`로 구현
- backend는 이미지 한 장을 받아 전처리 → 추론 → 후처리까지 수행한 뒤 `Detection` 결과를 반환
- 상위 코드는 OpenCV DNN이나 입력·출력 텐서 구조를 몰라도 되도록 구성
- backend 생성은 factory 함수로 통일
- 실행할 때 `--backend opencv_dnn` 옵션으로 backend를 선택할 수 있도록 추가
- Full + Crop 이중 추론 정책은 `YoloDetector`에 유지
- 중복되어 있던 NMS 코드를 하나로 정리
- 추론 시간을 전처리 / 추론 / 후처리로 나누어 `InferenceTiming`으로 받을 수 있도록 추가
- 기존 `[PERF]` 출력과 프로그램 동작은 그대로 유지
- 리팩터링 전후 결과가 동일한지 확인

### 이 작업을 한 이유
- 9월에 TensorRT를 붙일 때 기존 코드를 다시 뜯어고치지 않고 `TensorRTBackend`만 추가하기 위해
- OpenCV DNN과 TensorRT의 입력·출력 방식 차이를 backend 내부에서 처리하기 위해
- Logging Schema에서 전처리 / 추론 / 후처리 시간을 각각 기록하기 위해


### 2. 영상 입력 구조 분리
- `main.cpp`에서 직접 사용하던 `cv::VideoCapture`를 `VideoFileSource`로 분리
- 공통 인터페이스인 `FrameSource`를 만들고, `read()`가 `Frame`을 반환하도록 구성
- `Frame`에는 다음 정보를 저장
  - `image`: 실제 영상 프레임
  - `frameSeq`: 프레임 순번
  - `captureTimestampNs`: 프레임의 시간 정보
  - `captureTimestampSource`: 시간 정보의 출처
- 영상 파일에서는 `captureTimestampSource`를 `video_pts`로 사용
- 시간 단위는 이후 V4L2 카메라와 ROS2에서도 그대로 사용할 수 있도록 ns로 통일
- 영상 크기와 FPS 정보도 `FrameSource`를 통해 가져오도록 변경
- 기존 처리 루프는 거의 변경하지 않고 영상 입력 부분만 교체
- 리팩터링 전후 실행 결과가 동일한지 확인

### 이 작업을 한 이유
- 9월에 실제 카메라를 연결할 때 `VideoFileSource` 대신 `CameraSource`만 연결할 수 있도록 하기 위해
- 영상 파일과 실제 카메라가 같은 `Frame` 구조를 사용하도록 만들기 위해
- `capture_ts`와 이후 만들 `decision_ts`를 이용해 Frame Age를 계산하기 위해
- 8월 PC Baseline과 9월 Jetson 결과를 같은 로그 형식으로 분석하기 위해


### 알게 된 점
- 기존 `[PERF]`의 Post는 전체 후처리 시간이 아니라 일부 후처리 시간만 측정하고 있었음. 이번에 분리하면서 전처리 / 추론 / 후처리 시간을 따로 볼 수 있게 됨
- CMake에서 의존성을 `PRIVATE`로 설정해도 공개 헤더에 `cv::dnn::Net`, `cv::VideoCapture` 같은 타입이 들어 있으면 외부 코드에서도 그 기능을 알아야 함. 구현에만 필요한 타입은 헤더 밖으로 숨겨야 함


### 3. 실행 로그 스키마 확정 및 구현
- 실행 1회를 하나의 Run으로 관리하고 `results/runs/<run_id>/` 아래에 로그 생성
  - `raw_frame_log.csv`: 프레임별 처리시간과 FCW 상태
  - `run_summary.json`: 코드 버전, 입력, 모델, 실행 환경 등 Run의 조건
- 로그 작성 기능은 `logging/`의 `run_logger` 라이브러리로 분리
  - `adas`, `perception_demo`가 같은 스키마 사용
  - 두 앱의 공통 실행 옵션은 `apps/common/RunOptions.hpp`로 통합
- 실행 옵션 추가
  - `--run-id`
  - `--power-mode`
  - `--warmup-frames`
  - `--measured-frames`

#### 기록하는 값
- 프레임 식별: `frame`, `frame_seq`
  - 프로그램이 처리한 순서와 실제 입력 프레임 번호를 비교해서 Frame Drop이 있는지 확인
- Timestamp: `capture_ts`, `dequeue_ts`, `decision_ts`
  - `capture_ts`: 프레임이 입력된 시각
  - `dequeue_ts`: 프로그램이 프레임 처리를 시작한 시각
  - `decision_ts`: FCW 판단이 끝난 시각
  - `total_processing_ms = decision_ts - dequeue_ts`
  - 서로 다른 시간 기준을 잘못 계산하지 않도록 `capture_ts_clock`도 함께 기록
  - 영상 파일은 `stream` 기준이고, 실제 카메라는 `monotonic` 기준
  - 영상 파일은 실제 시스템 시간과 비교할 수 없으므로 Frame Age는 `n/a`
- 단계별 처리시간
  - Full / Crop 각각
    - `preprocess`
    - `inference`
    - `postprocess`
  - 추가로
    - `merge`
    - `detect`
    - `tracking`
    - `decision`
    - `output`
  - 단순히 "YOLO가 느리다"가 아니라 실제로 어느 단계가 오래 걸리는지 확인하기 위해 세분화
- FCW 상태
  - `detections`
  - `tracks`
  - `lead_id`
  - `lead_found`
  - `ttc_p`
  - `risk_state`
  - `warning_state`
  - `scene_changed`
  - LEAD 차량이 얼마나 안정적으로 유지되는지, TTC-P가 얼마나 자주 계산되는지 등을 확인하는 데 사용
- Run 정보
  - git commit / dirty 여부
  - hardware
  - backend
  - precision
  - build type
  - 입력 영상 / 모델 hash
  - 입력 영상 FPS (`source_fps`, 측정 FPS는 분석 스크립트가 계산)
  - confidence / NMS threshold
  - 파일 이름이 같더라도 실제 내용이 다를 수 있으므로 hash를 함께 기록

#### 분석 스크립트
- `tools/analyze_runs.py`
  - `raw_frame_log.csv`와 `run_summary.json`을 읽어서 성능을 자동으로 계산
  - FPS
  - p50 / p95 / max 처리시간
  - Deadline Miss Rate
  - Frame Age
  - Frame Drop
  - LEAD 유지 구간 수
  - LEAD 유지시간 중앙값
  - 짧게 끊기는 LEAD 구간 비율
  - TTC-P 산출률
  - 여러 Run을 비교할 경우 최소값 / 최대값 / 결과가 얼마나 흔들리는지도 계산
- `git_dirty: true`인 Run은 기본적으로 분석에서 제외
- 필요할 때만 `--include-dirty`를 사용
```bash
python3 tools/analyze_runs.py results/runs
```

#### 검증
- Logging Schema를 추가한 뒤에도 기존 FCW 결과가 바뀌지 않은 것 확인
  - 기존 프레임 결과 CSV가 이전과 동일
  - `perception_demo`가 만든 영상도 이전과 동일
- 기록된 시간 값이 정상적으로 계산되는 것 확인
  - 프레임 처리 시작부터 FCW 판단 완료까지의 시간이 올바르게 기록됨
  - 단계별 처리시간도 정상적으로 기록됨
- 기존에 기록하던 FCW 정보와 새 로그의 값이 동일한 것 확인
  - Detection 수
  - Track 수
  - LEAD 차량
  - 위험 상태
  - 장면 전환 등
- `--warmup-frames`, `--measured-frames` 옵션이 지정한 프레임 수대로 동작하는 것 확인
- 커밋하지 않은 코드로 실행하면 `git_dirty: true`,
  커밋 후 다시 빌드하면 `git_dirty: false`로 기록되는 것 확인
- `tools/analyze_runs.py`가 생성된 로그를 읽고 FPS, 처리시간, FCW 관련 지표를 정상적으로 계산하는 것 확인

### 알게 된 점
- 평가기준이 "무엇을 볼 것인가"라면 Logging Schema는 "그 값을 계산하기 위해 무엇을 남길 것인가"
- 최적화는 FPS만 오르는 게 아니라 FCW 판단 결과를 유지하면서 처리속도가 빨라져야 함
- p50은 평소 처리시간, p95는 가끔 느려질 때의 처리시간을 보는 값
- 영상 파일의 `capture_ts`는 영상 자체의 시간이므로 실제 시스템 시간과 비교할 수 없음
  - 따라서 영상 파일에서 Frame Age가 `n/a`인 것이 정상
- git 정보는 build할 때 갱신해야 실제 실행한 코드가 어느 commit인지 정확하게 남길 수 있음
- 성능 측정 순서
  - 코드 수정
  - 검증
  - commit
  - build
  - 측정
- raw 로그에는 실제 측정값을 그대로 남기고, 평균이나 p50 같은 통계값은 분석 스크립트에서 계산하는 것이 관리하기 편함
- 같은 PC에서 대용량 다운로드가 돌아가면 성능 측정값이 흔들릴 수 있으므로 Baseline 측정할 때는 다른 무거운 작업을 끄는 것이 좋음

### 다음 작업
- x86 PC에서 현재 코드의 기본 성능(Baseline) 측정
- 측정할 때 조건을 매번 똑같이 맞춤
  - 같은 입력 영상
  - 같은 YOLO 모델
  - 같은 confidence / NMS threshold
  - Release 빌드
  - 같은 warmup / measured 프레임 수
  - 회차 간 유휴 시간 동일 (CPU 클럭·온도 편차 통제)
  - 전원 연결 상태 동일, `--power-mode`로 기록
  - 측정 중 다운로드 등 다른 무거운 작업 중지
- 같은 조건으로 5번 반복 측정
  - `baseline_x86_r1`
  - `baseline_x86_r2`
  - `baseline_x86_r3`
  - `baseline_x86_r4`
  - `baseline_x86_r5`
- 5번의 결과가 원래 어느 정도씩 흔들리는지 확인
- 이후 코드를 최적화했을 때 이 범위를 확실히 넘어 성능이 좋아졌는지 비교
