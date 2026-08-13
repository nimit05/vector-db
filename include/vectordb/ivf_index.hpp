#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "vectordb/collection.hpp"

namespace vectordb
{

    class IVFIndex
    {
    public:
        std::vector<SearchResult> search(const std::unordered_map<std::string, VectorRecord> &records,
                                         const std::vector<double> &query,
                                         std::size_t k,
                                         std::size_t nprobe) const;
    };

}