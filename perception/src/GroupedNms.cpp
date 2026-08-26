#include "GroupedNms.hpp"

#include "perception/Classes.hpp"

#include <opencv2/dnn.hpp>
#include <map>
#include <vector>

std::vector<Detection> applyGroupedNms(const std::vector<Detection>& input, float scoreThreshold, float nmsThreshold) {
    std::map<int, std::vector<int>> indicesByGroup;
    for (int index = 0; index < static_cast<int>(input.size()); ++index) {
        indicesByGroup[getNmsGroup(input[index].classId)].push_back(index);
    }

    std::vector<Detection> result;
    for (const auto& groupEntry : indicesByGroup) {
        const std::vector<int>& globalIndices = groupEntry.second;
        std::vector<cv::Rect> groupBoxes;
        std::vector<float> groupScores;

        for (const int globalIndex : globalIndices) {
            groupBoxes.push_back(input[globalIndex].box);
            groupScores.push_back(input[globalIndex].confidence);
        }

        std::vector<int> keptIndices;
        cv::dnn::NMSBoxes(groupBoxes, groupScores, scoreThreshold, nmsThreshold, keptIndices);

        for (const int localIndex : keptIndices) {
            result.push_back(input[globalIndices[localIndex]]);
        }
    }
    return result;
}
