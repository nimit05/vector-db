#include <iostream>
#include <vector>

#include "vectordb/collection.hpp"
#include "vectordb/vectorRecord.hpp"

int main() {
    try {
        vectordb::Collection collection("demo");

        collection.insert(vectordb::VectorRecord("vec1", {1.0, 0.0, 0.0}));
        collection.insert(vectordb::VectorRecord("vec2", {0.0, 1.0, 0.0}));
        collection.insert(vectordb::VectorRecord("vec3", {0.8, 0.2, 0.0}));
        collection.insert(vectordb::VectorRecord("vec4", {0.7, 0.3, 0.0}));

        std::vector<double> query = {1.0, 0.0, 0.0};
        std::size_t k = 3;

        auto results = collection.search(query, k);

        std::cout << "Top " << k << " results for query:\n";
        for (const auto& result : results) {
            std::cout << "ID: " << result.id
                      << ", Score: " << result.score << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}