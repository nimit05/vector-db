#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "vectordb/vectorRecord.hpp"

namespace vectordb
{

    class IVFIndex;

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

        // Built lazily on the first IVF search and reused until the records
        // change. shared_ptr rather than unique_ptr because Collection is copied
        // and returned by value; a unique_ptr member would delete the copy
        // constructor.
        mutable std::shared_ptr<const IVFIndex> index_;
        mutable bool indexDirty_ = true;

        const IVFIndex &ivfIndex() const;

    public:
        explicit Collection(const std::string &name);

        const std::string &getName() const;
        void insert(const VectorRecord &record);
        bool remove(const std::string &id);

        void saveToFile(const std::string &filename) const;
        static Collection loadFromFile(const std::string &filename);

        std::vector<SearchResult> searchExact(const std::vector<double> &query, std::size_t k) const;

        /// Approximate top-k, scanning only the nprobe most promising clusters.
        /// Passing nprobe >= clusterCount() scans everything and matches searchExact.
        std::vector<SearchResult> searchIVF(const std::vector<double> &query, std::size_t k, std::size_t nprobe) const;

        /// Number of clusters in the index, building it if necessary.
        std::size_t clusterCount() const;
        std::vector<SearchResult> search(const std::vector<double> &query, std::size_t k) const;
        std::vector<VectorRecord> listRecords() const;
    };

}