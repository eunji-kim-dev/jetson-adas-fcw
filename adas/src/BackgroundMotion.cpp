#include "adas/BackgroundMotion.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>
#include <utility>
#include <cstdint>
#include <limits>

namespace {

// 아래 기준은 실험 초기값임. 회전 보정의 채택 여부는 양성·음성 영상 재검증으로 정함
constexpr int maximumWidth = 640;
constexpr int gridColumns = 6;
constexpr int gridRows = 3;
constexpr int cornersPerCell = 12;
constexpr int minimumSidePoints = 8;
constexpr int boxPadding = 12;              // 축소 영상 픽셀 단위임. LK 창이 객체 가장자리에 걸치는 것을 줄임
constexpr float maximumFlowError = 20.0F;
constexpr float maximumForwardBackwardError = 1.0F;
constexpr float stationaryMotion = 0.35F;
constexpr float minimumInlierRatio = 0.50F;
constexpr float maximumModelResidual = 1.0F;
constexpr int ransacTrials = 256;
constexpr float minimumTriangleArea = 0.005F; // 세 점의 외적 크기를 영상 면적 비율로 검사함
constexpr float minimumSideSpan = 0.06F;
constexpr float minimumTotalSpan = 0.40F;
constexpr float minimumMedianSeparation = 0.25F;
constexpr float minimumVerticalSpan = 0.08F;

struct FlowPoint {
    cv::Point2f position;
    float dx;
    float dy;
};

float median(std::vector<float> values) {
    if (values.empty()) return 0.0F;
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const float upper = values[middle];
    if (values.size() % 2 != 0) return upper;
    return (upper + *std::max_element(values.begin(), values.begin() + middle)) * 0.5F;
}

float medianComponent(const std::vector<FlowPoint>& points, bool horizontal) {
    std::vector<float> values;
    values.reserve(points.size());
    for (const auto& point : points) values.push_back(horizontal ? point.dx : point.dy);
    return median(std::move(values));
}

bool insideMask(const cv::Mat& mask, const cv::Point2f& point) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) return false;
    const int x = cvRound(point.x);
    const int y = cvRound(point.y);
    return x >= 2 && x < mask.cols - 2 && y >= 2 && y < mask.rows - 2
        && mask.at<unsigned char>(y, x) != 0;
}

void excludeBox(cv::Mat& mask, const cv::Rect& sourceBox, const cv::Size& sourceSize) {
    if (sourceBox.width <= 0 || sourceBox.height <= 0) return;
    const float scaleX = static_cast<float>(mask.cols) / static_cast<float>(sourceSize.width);
    const float scaleY = static_cast<float>(mask.rows) / static_cast<float>(sourceSize.height);
    const double right = static_cast<double>(sourceBox.x) + sourceBox.width;
    const double bottom = static_cast<double>(sourceBox.y) + sourceBox.height;
    const int x0 = static_cast<int>(std::floor(std::clamp(static_cast<double>(sourceBox.x), 0.0, static_cast<double>(sourceSize.width)) * scaleX));
    const int y0 = static_cast<int>(std::floor(std::clamp(static_cast<double>(sourceBox.y), 0.0, static_cast<double>(sourceSize.height)) * scaleY));
    const int x1 = static_cast<int>(std::ceil(std::clamp(right, 0.0, static_cast<double>(sourceSize.width)) * scaleX));
    const int y1 = static_cast<int>(std::ceil(std::clamp(bottom, 0.0, static_cast<double>(sourceSize.height)) * scaleY));
    const cv::Rect expanded(x0 - boxPadding, y0 - boxPadding,
                            std::max(0, x1 - x0) + boxPadding * 2,
                            std::max(0, y1 - y0) + boxPadding * 2);
    const cv::Rect clipped = expanded & cv::Rect(0, 0, mask.cols, mask.rows);
    if (!clipped.empty()) mask(clipped).setTo(0);
}

struct FlowPlane {
    double xSlope = 0.0;
    double ySlope = 0.0;
    double centerDx = 0.0;
};

// 수평 이동장을 dx=a(x-cx)+c(y-cy)+b로 근사함. 화면 중심의 b만 보정값으로 사용함
// x·y 기울기는 위치에 따른 확대·시차·수평 흐름 변화를 설명하는 항이며 차량에서 빼지 않음
// b도 실제 yaw 분리값은 아님. 확대 중심이 좌우 대표점 범위 밖이면 전진과 구분되지 않을 수 있음
double horizontalResidual(const FlowPoint& point, const FlowPlane& model, const cv::Point2f& center) {
    return std::abs(point.dx - (model.xSlope * (point.position.x - center.x)
        + model.ySlope * (point.position.y - center.y) + model.centerDx));
}

std::vector<FlowPoint> modelInliers(const std::vector<FlowPoint>& points, const FlowPlane& model,
                                   const cv::Point2f& center) {
    std::vector<FlowPoint> inliers;
    inliers.reserve(points.size());
    for (const auto& point : points) {
        if (horizontalResidual(point, model, center) <= maximumModelResidual) inliers.push_back(point);
    }
    return inliers;
}

double residualSum(const std::vector<FlowPoint>& points, const FlowPlane& model, const cv::Point2f& center) {
    double total = 0.0;
    for (const auto& point : points) total += horizontalResidual(point, model, center);
    return total;
}

// 세 계수의 최소제곱 문제를 부분 피벗 가우스 소거법으로 풂. 퇴화한 점 배치는 거부함
bool fitPlane(const std::vector<FlowPoint>& points, const cv::Point2f& center, FlowPlane& model) {
    double matrix[3][4]{};
    for (const auto& point : points) {
        const double v[3] = {point.position.x - center.x, point.position.y - center.y, 1.0};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) matrix[i][j] += v[i] * v[j];
            matrix[i][3] += v[i] * point.dx;
        }
    }
    for (int i = 0; i < 3; ++i) {
        int pivot = i;
        for (int j = i + 1; j < 3; ++j) {
            if (std::abs(matrix[j][i]) > std::abs(matrix[pivot][i])) pivot = j;
        }
        if (std::abs(matrix[pivot][i]) < 1.0e-6) return false;
        for (int j = i; j < 4; ++j) std::swap(matrix[i][j], matrix[pivot][j]);
        const double divisor = matrix[i][i];
        for (int j = i; j < 4; ++j) matrix[i][j] /= divisor;
        for (int row = 0; row < 3; ++row) {
            if (row == i) continue;
            const double multiplier = matrix[row][i];
            for (int j = i; j < 4; ++j) matrix[row][j] -= multiplier * matrix[i][j];
        }
    }
    model = {matrix[0][3], matrix[1][3], matrix[2][3]};
    return std::isfinite(model.xSlope) && std::isfinite(model.ySlope) && std::isfinite(model.centerDx);
}

float span(const std::vector<float>& values) {
    if (values.empty()) return 0.0F;
    const auto limits = std::minmax_element(values.begin(), values.end());
    return *limits.second - *limits.first;
}

// 기울기와 중심 이동을 구분할 수 있도록 화면 양쪽의 지지점과 전체 좌표 분포를 검사함
bool supportsPlane(const std::vector<FlowPoint>& points, std::size_t totalCount, int width, int height) {
    std::vector<float> leftX, rightX, allY;
    for (const auto& point : points) {
        (point.position.x < width * 0.5F ? leftX : rightX).push_back(point.position.x);
        allY.push_back(point.position.y);
    }
    if (leftX.size() < minimumSidePoints || rightX.size() < minimumSidePoints
        || points.size() < totalCount * minimumInlierRatio) return false;
    const auto leftLimits = std::minmax_element(leftX.begin(), leftX.end());
    const auto rightLimits = std::minmax_element(rightX.begin(), rightX.end());
    return span(leftX) >= width * minimumSideSpan && span(rightX) >= width * minimumSideSpan
        && *rightLimits.second - *leftLimits.first >= width * minimumTotalSpan
        && median(rightX) - median(leftX) >= width * minimumMedianSeparation
        && span(allY) >= height * minimumVerticalSpan;
}

bool betterModel(std::size_t count, double error, std::size_t bestCount, double bestError) {
    return count > bestCount || (count == bestCount && error < bestError);
}

} // namespace

void BackgroundMotionEstimator::reset() {
    previousGray_.release();
    previousMask_.release();
    previousSourceSize_ = cv::Size();
}

BackgroundMotionResult BackgroundMotionEstimator::update(
    const cv::Mat& frame, const std::vector<Detection>& detections,
    const std::vector<TrackedObject>& trackedObjects) {
    BackgroundMotionResult result;
    if (frame.empty() || frame.depth() != CV_8U
        || (frame.channels() != 1 && frame.channels() != 3 && frame.channels() != 4)) {
        reset();
        result.state = "input";
        return result;
    }

    const int width = std::min(maximumWidth, frame.cols);
    const int height = std::max(1, cvRound(static_cast<double>(frame.rows) * width / frame.cols));
    if (width < 32 || height < 32) {
        reset();
        result.state = "input";
        return result;
    }

    cv::Mat small;
    cv::resize(frame, small, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
    cv::Mat gray;
    if (small.channels() == 1) gray = small;
    else cv::cvtColor(small, gray, small.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);

    cv::Mat mask;
    cv::threshold(gray, mask, 8, 255, cv::THRESH_BINARY);
    cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 1);
    const int bandTop = cvRound(height * 0.05);
    const int bandBottom = cvRound(height * 0.60);
    mask.rowRange(0, bandTop).setTo(0);
    mask.rowRange(bandBottom, height).setTo(0);
    for (const auto& detection : detections) excludeBox(mask, detection.box, frame.size());
    for (const auto& track : trackedObjects) excludeBox(mask, track.box, frame.size());

    if (previousGray_.empty() || previousSourceSize_ != frame.size() || previousGray_.size() != gray.size()) {
        result.state = previousGray_.empty() ? "first" : "size";
        previousGray_ = gray;
        previousMask_ = mask;
        previousSourceSize_ = frame.size();
        return result;
    }

    // 각 격자에서 따로 고르게 뽑아 한쪽 배경의 강한 무늬가 전체 추정을 독점하지 않게 함
    std::vector<cv::Point2f> previousPoints;
    for (int row = 0; row < gridRows; ++row) {
        const int y0 = bandTop + (bandBottom - bandTop) * row / gridRows;
        const int y1 = bandTop + (bandBottom - bandTop) * (row + 1) / gridRows;
        for (int col = 0; col < gridColumns; ++col) {
            const int x0 = width * col / gridColumns;
            const int x1 = width * (col + 1) / gridColumns;
            const cv::Rect cell(x0, y0, x1 - x0, y1 - y0);
            std::vector<cv::Point2f> corners;
            cv::goodFeaturesToTrack(previousGray_(cell), corners, cornersPerCell, 0.01, 7.0,
                                    previousMask_(cell), 3, false, 0.04);
            for (auto point : corners) {
                point += cv::Point2f(static_cast<float>(x0), static_cast<float>(y0));
                previousPoints.push_back(point);
            }
        }
    }

    std::vector<FlowPoint> left, right;
    if (!previousPoints.empty()) {
        std::vector<cv::Point2f> nextPoints;
        std::vector<unsigned char> status;
        std::vector<float> error;
        const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
        cv::calcOpticalFlowPyrLK(previousGray_, gray, previousPoints, nextPoints, status, error,
                                cv::Size(21, 21), 3, criteria);

        std::vector<cv::Point2f> forwardPrevious, forwardCurrent;
        for (std::size_t i = 0; i < previousPoints.size(); ++i) {
            if (!status[i] || !std::isfinite(error[i]) || error[i] > maximumFlowError
                || !insideMask(previousMask_, previousPoints[i]) || !insideMask(mask, nextPoints[i])) continue;
            forwardPrevious.push_back(previousPoints[i]);
            forwardCurrent.push_back(nextPoints[i]);
        }
        if (!forwardCurrent.empty()) {
            std::vector<cv::Point2f> backwardPoints;
            std::vector<unsigned char> backwardStatus;
            std::vector<float> backwardError;
            cv::calcOpticalFlowPyrLK(gray, previousGray_, forwardCurrent, backwardPoints,
                                    backwardStatus, backwardError, cv::Size(21, 21), 3, criteria);
            for (std::size_t i = 0; i < forwardCurrent.size(); ++i) {
                if (!backwardStatus[i] || !std::isfinite(backwardError[i]) || backwardError[i] > maximumFlowError
                    || !insideMask(previousMask_, backwardPoints[i])) continue;
                const cv::Point2f fb = backwardPoints[i] - forwardPrevious[i];
                if (fb.dot(fb) > maximumForwardBackwardError * maximumForwardBackwardError) continue;
                const cv::Point2f flow = forwardCurrent[i] - forwardPrevious[i];
                const FlowPoint point{forwardPrevious[i], flow.x, flow.y};
                if (point.position.x < width * 0.5F) left.push_back(point);
                else right.push_back(point);
            }
        }
    }

    // 추정 실패 프레임도 다음 추적의 기준 영상으로 삼음. 유효하지 않은 이동을 여러 프레임에 걸쳐 누적하지 않음
    previousGray_ = gray;
    previousMask_ = mask;
    previousSourceSize_ = frame.size();

    const float scaleBack = static_cast<float>(frame.cols) / static_cast<float>(width);
    const cv::Point2f center(width * 0.5F, height * 0.5F);
    std::vector<FlowPoint> points = left;
    points.insert(points.end(), right.begin(), right.end());
    if (left.size() < minimumSidePoints || right.size() < minimumSidePoints) {
        result.state = "few";
        return result;
    }

    // 같은 입력에서 같은 후보를 검사하도록 고정 시드로 좌·우·전체에서 세 점을 고름
    std::uint32_t seed = 12943;
    const auto next = [&seed]() {
        seed = 1664525U * seed + 1013904223U;
        return seed;
    };
    FlowPlane bestModel;
    std::vector<FlowPoint> bestInliers;
    double bestError = std::numeric_limits<double>::infinity();
    for (int trial = 0; trial < ransacTrials; ++trial) {
        const FlowPoint l = left[next() % left.size()];
        const FlowPoint r = right[next() % right.size()];
        const FlowPoint third = points[next() % points.size()];
        const cv::Point2f u = r.position - l.position;
        const cv::Point2f v = third.position - l.position;
        if (std::abs(u.x * v.y - u.y * v.x) < width * height * minimumTriangleArea) continue;
        FlowPlane candidateModel;
        if (!fitPlane({l, r, third}, center, candidateModel)) continue;
        auto candidate = modelInliers(points, candidateModel, center);
        if (!supportsPlane(candidate, points.size(), width, height)) continue;
        double error = residualSum(candidate, candidateModel, center);
        for (int refine = 0; refine < 2; ++refine) {
            FlowPlane refinedModel;
            if (!fitPlane(candidate, center, refinedModel)) break;
            auto refined = modelInliers(points, refinedModel, center);
            if (!supportsPlane(refined, points.size(), width, height)) break;
            const double refinedError = residualSum(refined, refinedModel, center);
            // 재계산이 지지점·잔차 기준을 개선할 때만 채택함. 검증된 모델을 정제 과정에서 잃지 않음
            if (!betterModel(refined.size(), refinedError, candidate.size(), error)) break;
            candidate = std::move(refined);
            candidateModel = refinedModel;
            error = refinedError;
        }
        if (betterModel(candidate.size(), error, bestInliers.size(), bestError)) {
            bestInliers = std::move(candidate);
            bestModel = candidateModel;
            bestError = error;
        }
    }
    if (bestInliers.empty()) {
        result.state = "model";
        return result;
    }
    std::vector<FlowPoint> leftInliers, rightInliers;
    for (const auto& point : bestInliers) {
        (point.position.x < center.x ? leftInliers : rightInliers).push_back(point);
    }
    result.leftCount = static_cast<int>(leftInliers.size());
    result.rightCount = static_cast<int>(rightInliers.size());
    result.leftDxPixels = medianComponent(leftInliers, true) * scaleBack;
    result.rightDxPixels = medianComponent(rightInliers, true) * scaleBack;
    const float largestDy = std::max(std::abs(medianComponent(leftInliers, false)),
                                     std::abs(medianComponent(rightInliers, false)));
    if (largestDy > std::max(0.5F, static_cast<float>(std::abs(bestModel.centerDx)) * 0.5F)) {
        result.state = "vertical";
        return result;
    }
    if (std::abs(bestModel.centerDx) > width * 0.08F) {
        result.state = "large";
        return result;
    }
    // 중심 이동이 유의미한데 양측 대표 흐름이 반대이면 확대와 구별되지 않아 보정하지 않음
    if (std::abs(bestModel.centerDx) > stationaryMotion && result.leftDxPixels * result.rightDxPixels <= 0.0F) {
        result.state = "direction";
        return result;
    }
    result.valid = true;
    // zero는 중심의 공통 수평 성분이 작다는 뜻임. 화면 전체가 정지했다는 뜻은 아님
    if (std::abs(bestModel.centerDx) <= stationaryMotion) {
        result.state = "zero";
    } else {
        result.state = "valid";
        result.dxPixels = static_cast<float>(bestModel.centerDx) * scaleBack;
    }
    return result;
}