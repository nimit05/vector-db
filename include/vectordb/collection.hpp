#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "vectordb/vectorRecord.hpp"

namespace vectordb
{

    class IVFIndex;
    struct SearchResponse;

    // Declared, not defined, to keep this header free of ivf_index.hpp, which
    // includes this one for SearchResult. The definition in ivf_index.hpp must
    // keep the same (default) underlying type.
    enum class SearchMode;

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

        /// Exact top-k through the IVF index. Stops as soon as the result is
        /// provably exact; the response says whether it stopped early (Certified)
        /// or scanned every cluster (Exhausted), and how much it scanned.
        SearchResponse searchIVF(const std::vector<double> &query, std::size_t k) const;

        /// Top-k under a mode's latency contract. SearchMode::Exact is
        /// guaranteed exact; the other modes trade recall for latency and report
        /// in the status how much of the index they got through.
        SearchResponse searchIVF(const std::vector<double> &query, std::size_t k, SearchMode mode) const;

        /// Approximate top-k, scanning only the nprobe most promising clusters.
        /// Passing nprobe >= clusterCount() scans everything and matches searchExact.
        std::vector<SearchResult> searchIVF(const std::vector<double> &query, std::size_t k, std::size_t nprobe) const;

        /// Number of clusters in the index, building it if necessary.
        std::size_t clusterCount() const;
        std::vector<SearchResult> search(const std::vector<double> &query, std::size_t k) const;
        std::vector<VectorRecord> listRecords() const;
    };

}