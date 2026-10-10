# 분석 문서 — Q1~Q6

9/5 디렉터용 계획서에 적은 질문 6개를 실험 결과와 함께 정리했다. 질문별 최종 문서와 함께, 이전에 작성한 기록도 남겼다. 이전 기록에서 달라진 내용은 각 문서 맨 위에 정정 사항으로 표시했다.

기준일: 2026-10-10

---

## 질문별 답

| 질문 | 핵심 결과 | 최종 문서 | 기록 문서 |
| --- | --- | --- | --- |
| **Q1** FCW가 제대로 동작하는가 | 본 세트 16개에서 Miss 1(01 자차 회전), 정상 주행 10개에서 오경보 0. 경고 선행시간은 평균 1.11초였다. 처음 보는 영상 10개에서는 놓침 1·오경보 1이 나왔다. | [q1-fcw-final-result.md](q1-fcw-final-result.md) | [q1-ttc-instability.md](q1-ttc-instability.md) — 9/13 TTC-P 1픽셀 문제, 9/24 21개 1차 결과 |
| **Q2** 얼마나 빠른가 | Jetson CPU 237 ms(4.2 FPS) → GPU 전처리 16.7 ms(46.8 FPS) → GPU 후처리까지 11.2 ms(63.4 FPS). 세 경로를 같은 밤, 같은 빌드로 연이어 5회씩 측정했다. 최종 처리 시간은 목표 66.7 ms의 0.17배다. 카메라는 처리 9.5 ms·Frame Age 41.7 ms였다. | [q2-q3-speed-bottleneck.md](q2-q3-speed-bottleneck.md) | [q2-q3-jetson-cpu-baseline.md](q2-q3-jetson-cpu-baseline.md) — 9/13, 최적화 없는 빌드 값 |
| **Q3** 어디서 느린가 | 처음에는 YOLO 추론이 병목이었고, 이후 전처리·후처리를 거쳐 다시 추론이 가장 큰 비중을 차지했다(현재 69%). 카메라 Frame Age는 카메라 자체 지연 32 ms에 처리 시간을 더한 값이다. | [q2-q3-speed-bottleneck.md](q2-q3-speed-bottleneck.md) | 위와 같음 |
| **Q4** 무엇을 바꾸면 빨라지는가 | 효과는 TensorRT 가속(약 6배) > Crop 288(−20%) > FP16(−16.5%) > INT8(Entropy −8% 탈락, MinMax −16% 보류) 순이었다. 이후에는 정밀도와 무관한 고정 비용이 더 큰 비중을 차지했다. | [q4-structure-precision-speed.md](q4-structure-precision-speed.md) | — |
| **Q5** 그 대신 정확도는 얼마나 떨어지는가 | FP16·GPU 전처리·GPU 후처리는 FCW 결과가 같았다. Crop 288은 21개 중 20개가 첫 경고까지 같았고, 04만 달랐다(640 놓침 → 288 경고). INT8 Entropy는 검출 25% 손실이 있었고, MinMax는 좌표 흔들림의 영향을 검증하지 못해 보류했다. | [q5-precision-accuracy.md](q5-precision-accuracy.md) | [q5-precision-fcw-impact.md](q5-precision-fcw-impact.md) — 9/18 input.mp4 한 편 |
| **Q6** 제한된 HW에서 어떤 설정이 좋은가 | Full 640 + Crop 288 / TensorRT FP16 / GPU 전처리·후처리 / 동기 캡처 / LEAD 12규칙 / MAXN_SUPER를 선택했다. 보드 전체 RAM은 약 2.2 GB(8 GB 보드)였고, 파일 프레임당 에너지는 276 → 180 mJ로 줄었다. | [q6-embedded-configuration.md](q6-embedded-configuration.md) | — |

---

## 측정 조건과 수치 기준

- 장비는 Jetson Orin Nano Super (JetPack 7, CUDA 13.2, TensorRT 10.16)다. MAXN_SUPER 모드에서 `jetson_clocks`로 클럭을 고정했다.
- 속도는 `videos/input.mp4`(720p, 1252프레임)를 `scripts/benchmark_harness.sh`로 5회 측정한 p50 기준이다. 표에는 5회 범위를, 본문의 대표값에는 5회 가운데값을 썼다. 한 번만 측정한 값은 "1회", INT8 MinMax·Power Mode 표는 "3회 가운데"로 따로 표시했다. 파일 속도는 모두 LEAD 규칙을 끈 상태에서 측정했다.
- FCW 수치는 Jetson 실행 결과를 기준으로 했다. x86 결과는 규칙이 동작하는지 확인하는 데 사용했다. 두 장비에서 검출 박스가 조금씩 달라 경고 프레임도 달라질 수 있다.
- 9/29 전 속도에는 초록 도로 영역을 그리는 시간 약 3.7 ms가 포함돼 있다. 해당 값은 당시 측정끼리의 상대 비교에만 사용했다.

---

## 결과 재현 기준 — 골든 MD5 (`input.mp4`)

| 플랫폼 · 설정 | `_frames.csv` | `_banner.csv` |
| --- | --- | --- |
| x86 / opencv_dnn | `a0006c4a` | — |
| x86 / opencv_dnn + Crop 288 | `87453f9c` | — |
| x86 / opencv_dnn + Crop 288 + 12규칙 | `d5cf68be` | `c6e1c299` |
| Jetson / TensorRT FP16 + Crop 288 | `4681f432` | `e1a66eaf` |
| Jetson / TensorRT FP16 + Crop 288 + 12규칙 | `f72e9951` | `5e7da573` |

GPU 전처리·GPU 후처리 경로의 결과는 CPU 경로와 비트 단위로 같다. 경고 배너만 바꾸는 규칙(`rawconfirm` 등)은 `_frames.csv`만으로 변경 여부를 확인할 수 없어서, 12규칙은 `_banner.csv`도 함께 보관했다.

---

## 남은 작업

- 최종 설정으로 데모 영상 생성 (평가셋 41개 + `input.mp4`) — 10/10
- 1080p(03) GPU 후처리 속도 측정 — 아직 측정하지 않음 (선택)
