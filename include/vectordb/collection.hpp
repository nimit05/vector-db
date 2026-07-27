#pragma once

#include <string>
#include <unordered_map>
#include "vectordb/vectorRecord.hpp"

namespace vectordb
{

    struct SearchResult
    {
        std::string id;
        double score;
    };

    class Collection
    {
    private:
        std::string name_;
        std::unordered_map<std::string, VectorRecord> records_;

    public:
        explicit Collection(const std::string &name);

        const std::string &getName() const;
        void insert(const VectorRecord &record);
        bool remove(const std::string &id);

        void saveToFile(const std::string &filename) const;
        static Collection loadFromFile(const std::string &filename);

        std::vector<SearchResult> search(const std::vector<double> &query, std::size_t k) const;
        std::vector<VectorRecord> listRecords() const;
    };

}