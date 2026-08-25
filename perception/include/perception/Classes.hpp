#pragma once

#include <string>

// COCO 클래스 ID 기준의 검출 대상/차량 분류 체계
// perception과 adas가 공유하는 클래스 유틸리티
inline bool isTargetClass(int classId) {
    return classId == 0 || classId == 1 || classId == 2 || classId == 3 || classId == 5 || classId == 7;
}

inline bool isVehicleClass(int classId) {
    return classId == 2 || classId == 3 || classId == 5 || classId == 7;
}

inline std::string getClassName(int classId) {
    switch (classId) {
        case 0: return "person";
        case 1: return "bicycle";
        case 2: return "car";
        case 3: return "motorcycle";
        case 5: return "bus";
        case 7: return "truck";
        default: return "unknown";
    }
}

// NMS는 정확한 클래스가 아니라 비슷한 클래스 그룹 단위로 수행
// car/bus/truck 사이의 클래스 흔들림을 같은 후보로 취급
inline int getNmsGroup(int classId) {
    if (classId == 2 || classId == 5 || classId == 7) return 0;
    if (classId == 1 || classId == 3) return 1;
    if (classId == 0) return 2;
    return classId + 10;
}
