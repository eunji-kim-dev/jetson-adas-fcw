#include "adas/RoadHomography.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace {

std::string pointText(const cv::Point2d& point) {
    std::ostringstream stream;
    stream << '(' << point.x << ',' << point.y << ')';
    return stream.str();
}

} // namespace

bool RoadHomography::create(const std::array<cv::Point2d, 4>& imagePoints, double roadWidthM, double roadLengthM,
                            int frameWidth, int frameHeight, int bonnetY, RoadHomography& out, std::string& error) {
    if (!(roadWidthM > 0.0) || !(roadLengthM > 0.0)) {
        error = "노면 폭·길이(m)는 0 보다 커야 함";
        return false;
    }
    if (frameWidth <= 0 || frameHeight <= 0) {
        error = "영상 크기가 0 임";
        return false;
    }

    // 4점이 화면 안에 있어야 함. 벗어나면 라벨과 영상이 안 맞는 신호임
    for (const cv::Point2d& point : imagePoints) {
        if (!(point.x >= 0.0 && point.x <= frameWidth - 1 && point.y >= 0.0 && point.y <= frameHeight - 1)) {
            error = "노면 점 " + pointText(point) + " 이 영상 크기 " + std::to_string(frameWidth) + "x" + std::to_string(frameHeight) + " 를 벗어남";
            return false;
        }
    }

    const cv::Point2d& nearLeft = imagePoints[0];
    const cv::Point2d& nearRight = imagePoints[1];
    const cv::Point2d& farRight = imagePoints[2];
    const cv::Point2d& farLeft = imagePoints[3];

    // 순서 점검: 가까운 줄(P1·P2)이 먼 줄(P3·P4)보다 아래(y 가 큼), 각 줄에서 왼쪽 x < 오른쪽 x
    // 기울어진 4점(19)도 이 조건은 만족함
    if (!(std::min(nearLeft.y, nearRight.y) > std::max(farRight.y, farLeft.y))) {
        error = "가까운 줄(P1·P2)이 먼 줄(P3·P4)보다 아래여야 함 — 순서는 가까운 왼쪽, 가까운 오른쪽, 먼 오른쪽, 먼 왼쪽";
        return false;
    }
    if (!(nearLeft.x < nearRight.x) || !(farLeft.x < farRight.x)) {
        error = "각 줄의 왼쪽 점 x 가 오른쪽 점 x 보다 작아야 함";
        return false;
    }
    if (bonnetY >= frameHeight) {
        error = "보닛 선 y " + std::to_string(bonnetY) + " 가 영상 높이 " + std::to_string(frameHeight) + " 를 벗어남";
        return false;
    }

    const cv::Point2f source[4] = {
        cv::Point2f(static_cast<float>(nearLeft.x), static_cast<float>(nearLeft.y)),
        cv::Point2f(static_cast<float>(nearRight.x), static_cast<float>(nearRight.y)),
        cv::Point2f(static_cast<float>(farRight.x), static_cast<float>(farRight.y)),
        cv::Point2f(static_cast<float>(farLeft.x), static_cast<float>(farLeft.y)),
    };
    const cv::Point2f target[4] = {
        cv::Point2f(0.0F, 0.0F),
        cv::Point2f(static_cast<float>(roadWidthM), 0.0F),
        cv::Point2f(static_cast<float>(roadWidthM), static_cast<float>(roadLengthM)),
        cv::Point2f(0.0F, static_cast<float>(roadLengthM)),
    };

    RoadHomography result;
    const cv::Mat transform = cv::getPerspectiveTransform(source, target);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result.imageToRoad_(row, column) = transform.at<double>(row, column);
        }
    }

    // 4점의 동차 좌표 w 부호가 같아야 함 (볼록 사각형). 다르면 지평선이 4점 사이를 지나는 것 = 순서나 좌표가 틀림
    double sign = 0.0;
    for (const cv::Point2f& point : source) {
        const double w = result.imageToRoad_(2, 0) * point.x + result.imageToRoad_(2, 1) * point.y + result.imageToRoad_(2, 2);
        if (!std::isfinite(w) || std::abs(w) < 1.0e-12) {
            error = "노면 4점으로 변환표를 못 만듦 (세 점이 한 줄에 있거나 점이 겹침)";
            return false;
        }
        const double pointSign = w > 0.0 ? 1.0 : -1.0;
        if (sign == 0.0) {
            sign = pointSign;
        } else if (pointSign != sign) {
            error = "노면 4점이 볼록 사각형이 아님 (지평선이 4점 사이를 지남)";
            return false;
        }
    }
    result.roadSideSign_ = sign;

    // 왕복 점검: 4점을 다시 바꿨을 때 도로 좌표 (0,0)·(W,0)·(W,L)·(0,L) 과 1 mm 안으로 맞아야 함
    for (int k = 0; k < 4; ++k) {
        const cv::Vec3d mapped = result.imageToRoad_ * cv::Vec3d(source[k].x, source[k].y, 1.0);
        const double x = mapped[0] / mapped[2];
        const double z = mapped[1] / mapped[2];
        if (!(std::abs(x - target[k].x) < 1.0e-3 && std::abs(z - target[k].y) < 1.0e-3)) {
            error = "노면 4점 왕복 오차가 큼 (변환표 계산 실패)";
            return false;
        }
    }

    // 기준 깊이 Z_c: 화면 가로 가운데, 보닛 선(없으면 맨 아래 줄)
    result.bonnetY_ = bonnetY >= 0 ? bonnetY : -1;
    result.referenceRow_ = bonnetY >= 0 ? bonnetY : frameHeight - 1;
    const double centerX = (static_cast<double>(frameWidth) - 1.0) / 2.0;
    if (!result.depthAt(centerX, static_cast<double>(result.referenceRow_), result.referenceDepthM_)) {
        error = "기준점(화면 가운데, y " + std::to_string(result.referenceRow_) + ")이 지평선 위로 계산됨 — 노면 4점 확인 필요";
        return false;
    }

    out = result;
    return true;
}

bool RoadHomography::depthAt(double u, double v, double& depthM) const {
    // 동차 좌표 w 를 노면 쪽 부호로 맞춤. 0 이하면 지평선 위(또는 지평선)
    const double w = (imageToRoad_(2, 0) * u + imageToRoad_(2, 1) * v + imageToRoad_(2, 2)) * roadSideSign_;
    if (!(w > 1.0e-12)) return false;
    const double z = (imageToRoad_(1, 0) * u + imageToRoad_(1, 1) * v + imageToRoad_(1, 2)) * roadSideSign_ / w;
    if (!std::isfinite(z) || z > maximumDepthM) return false;
    depthM = z;
    return true;
}