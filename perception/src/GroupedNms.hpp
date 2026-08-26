#pragma once

#include "perception/Detection.hpp"

#include <vector>

/*
 * 클래스 그룹(car/bus/truck, bicycle/motorcycle, person) 단위 IoU NMS
 *
 * 정확한 클래스가 아니라 getNmsGroup() 기준으로 묶어서
 * car / bus / truck 사이의 클래스 흔들림을 같은 후보로 취급한다.
 *
 * 입력 순서를 그룹별로 보존하고, 결과는 그룹 번호 오름차순 →
 * 그룹 내 NMS 유지 순서로 반환한다.
 *
 * scoreThreshold는 cv::dnn::NMSBoxes 규칙을 따른다:
 * confidence > scoreThreshold 인 후보만 NMS 대상이 된다 (같으면 탈락).
 *
 * perception 계층에서 cv::dnn 심볼을 직접 사용하는 곳은
 * 이 함수의 구현부(GroupedNms.cpp)와 OpenCVDNNBackend뿐이다.
 */
std::vector<Detection> applyGroupedNms(const std::vector<Detection>& input, float scoreThreshold, float nmsThreshold);
