#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <string>

/*
 * --ttc-mode homography|both: 노면 4점으로 화면 픽셀을 도로 평면 좌표(가로 X, 깊이 Z, 단위 m)로 바꾸는 변환표 H
 *
 * 4점 순서 (eval/videos.csv 의 rx1..ry4): 가까운 줄 왼쪽 → 가까운 줄 오른쪽 → 먼 줄 오른쪽 → 먼 줄 왼쪽
 * 도로 좌표: P1 = (0, 0), P2 = (W, 0), P3 = (W, L), P4 = (0, L). 가까운 노면 줄이 Z = 0, 앞쪽이 +
 * W·L(폭·길이 가정)은 m 눈금만 정하고 TTR-H 시간 식에서는 약분됨. 시간에 영향을 주는 건 지평선(소실선) 위치뿐임 (5-2)
 * 단 아래 maximumDepthM 은 m 로 자르므로 L 가정이 "어디까지 깊이를 재는가"에는 영향을 줌 (L 을 2배로 두면 같은 픽셀이 2배 깊이 → 더 넓은 띠가 제외됨)
 *
 * 기준 깊이 Z_c: 화면 가로 가운데 x = (frameWidth − 1) / 2 에서 보닛 선(bonnet_y, 없으면 화면 맨 아래 줄)의 점 하나를 H 로 바꾼 Z
 * 가까운 노면 줄보다 아래라 보통 음수임. 기울어진 영상에서 같은 y 라도 Z 가 다르다는 건 알고 한 점으로 고정함
 */
class RoadHomography {
public:
    // 변환표를 만들고 점검함. 실패하면 false 와 이유(error)
    // 점검: 폭·길이 > 0, 4점이 화면 안, 가까운 줄이 먼 줄보다 아래, 왼쪽 x < 오른쪽 x,
    //       4점이 지평선 같은 쪽(볼록 사각형), 4점 왕복 오차, 보닛 선이 화면 안, 기준점이 지평선 아래
    // bonnetY: 보닛 선 y. 없으면 -1 (화면 맨 아래 줄을 기준으로 씀)
    static bool create(const std::array<cv::Point2d, 4>& imagePoints, double roadWidthM, double roadLengthM,
                       int frameWidth, int frameHeight, int bonnetY, RoadHomography& out, std::string& error);

    // 화면 점 (u, v) 의 깊이 Z (m). 지평선 위·근처(maximumDepthM 초과)면 false
    bool depthAt(double u, double v, double& depthM) const;

    double referenceDepthM() const { return referenceDepthM_; }   // Z_c
    int referenceRow() const { return referenceRow_; }            // Z_c 를 잡은 줄 (보닛 선 또는 맨 아래 줄)
    int bonnetY() const { return bonnetY_; }                      // 없으면 -1

    // 지평선 근처 보호. 이 깊이를 넘는 점은 지평선으로 봄 (depthAt 이 false → h_state=horizon)
    // 지평선 몇 px 아래는 1px 차이로 거리가 수십 m 바뀜. 300 m 앞 차는 높이 몇 px 라 경고 최소 높이(36px)에 한참 못 미침 (계산 보호용 값)
    // 제외되는 띠의 폭은 영상마다 다름 (L=10 기준 지평선 아래 4~33px. 13 은 y 475~508, 17 은 551~558, 19 는 688~693)
    static constexpr double maximumDepthM = 300.0;

private:
    cv::Matx33d imageToRoad_ = cv::Matx33d::eye();
    double roadSideSign_ = 1.0;   // 노면 4점에서의 동차 좌표 w 부호. 이 부호가 아닌 쪽이 지평선 위
    double referenceDepthM_ = 0.0;
    int referenceRow_ = -1;
    int bonnetY_ = -1;
};