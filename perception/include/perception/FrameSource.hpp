#pragma once

#include "perception/Frame.hpp"

/*
 * 프레임 입력 계약: "다음 프레임 하나를 메타데이터와 함께 준다"
 *
 * 영상 파일(VideoFileSource)과 실시간 카메라가 같은 인터페이스를 쓴다.
 * 카메라 구현이 밀린 프레임을 내부에서 버리는 정책(latest-frame)을
 * 갖더라도 이 계약은 바뀌지 않는다. 버려진 프레임은 frameSeq가
 * 건너뛰는 것으로 드러난다.
 */
class FrameSource {
public:
    virtual ~FrameSource() = default;

    // 프레임을 하나 읽어 frame에 채운다. 스트림 끝이나 실패면 false
    virtual bool read(Frame& frame) = 0;

    // 소스의 명목 해상도와 fps (영상 파일: 컨테이너 값, 카메라: 설정한 포맷)
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual double fps() const = 0;
};
