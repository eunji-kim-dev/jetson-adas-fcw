# 개발 작업일지

## 2026-08-24

### 진행한 작업 및 확인 결과
- ONNX opset 12 확인
- input: `images [1, 3, 640, 640]`
- output: `output0 [1, 84, 8400]`
- static shape 확인
- NMS는 ONNX 내부가 아니라 C++ `NMSBoxes()`에서 처리됨

### 알게 된 점
- 현재 ONNX 자체가 640×640 고정 입력으로 export되어 있음
- 960 입력을 비교하려면 960용 ONNX를 별도로 export해야 함

### 다음 작업
- 640 1회 / 640 + crop 2회 / 960 1회 비교