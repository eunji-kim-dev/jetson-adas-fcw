#pragma once
#include <optional>

// Jetson 전용 실행 환경 값. 지금은 전 플랫폼에서 nullopt → run_summary에 null로 기록.
// 9월 Jetson 세션에서 이 파일만 구현:
//  - readSocTemperatureC(): /sys/class/thermal/thermal_zone*/temp 최댓값 (밀리°C → °C)
//  - readJetsonClocksActive(): jetson_clocks 적용 상태
namespace jetson_env {

inline std::optional<double> readSocTemperatureC()    { return std::nullopt; }
inline std::optional<bool>   readJetsonClocksActive() { return std::nullopt; }

}  // namespace jetson_env