#include "vectordb/similarity.hpp"

#include <cmath>
#include <stdexcept>

namespace vectordb {

double cosineSimilarity(const std::vector<double>& a,
                        const std::vector<double>& b) {
    if (a.empty() || b.empty()) {
        throw std::invalid_argument("Vectors cannot be empty");
    }

    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same dimension");
    }

    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    if (normA == 0.0 || normB == 0.0) {
        throw std::invalid_argument("Zero vector is not allowed");
    }

    return dotProduct / (std::sqrt(normA) * std::sqrt(normB));
}

}