#pragma once

#include <vector>

namespace vectordb {

double cosineSimilarity(const std::vector<double>& a,
                        const std::vector<double>& b);

}