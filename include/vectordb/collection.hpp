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

    // Only declared here; the full enum is in ivf_index.hpp. (ivf_index.hpp
    // includes this file, so this file cannot include it back.)
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
        std::unordered_map<std::string, VectorRecord> records_; // id -> record

        // The IVF index is built on the first IVF search. After that, inserts
        // and removes update it directly. Once more than 20% of the collection
        // has changed since the last build, it is marked dirty and rebuilt on
        // the next IVF search.
        //
        // These are "mutable" because ivfIndex() is const (searching does not
        // change the records) but it still has to build and cache the index.
        mutable std::shared_ptr<IVFIndex> index_;
        mutable bool indexDirty_ = true;
        mutable std::size_t changesSinceBuild_ = 0;
        mutable std::size_t sizeAtBuild_ = 0;

        static constexpr double kRebuildChurn = 0.2; // rebuild after 20% changes

        // Returns the index, building it first if it is missing or dirty.
        const IVFIndex &ivfIndex() const;

        // Counts one change. Returns the index to update, or nullptr if there
        // is no index yet or it now needs a full rebuild.
        IVFIndex *indexForUpdate();

    public:
        explicit Collection(const std::string &name);

        const std::string &getName() const;
        void insert(const VectorRecord &record);
        bool remove(const std::string &id);

        void saveToFile(const std::string &filename) const;
        static Collection loadFromFile(const std::string &filename);

        // Brute force: scores every vector. Always exact.
        std::vector<SearchResult> searchExact(const std::vector<double> &query, std::size_t k) const;

        // IVF search, always exact. The response also says how much was scanned.
        SearchResponse searchIVF(const std::vector<double> &query, std::size_t k) const;

        // IVF search with a mode (LowLatency, Balanced, HighRecall, Exact).
        SearchResponse searchIVF(const std::vector<double> &query, std::size_t k, SearchMode mode) const;

        // IVF search that scans at most nprobe clusters.
        std::vector<SearchResult> searchIVF(const std::vector<double> &query, std::size_t k, std::size_t nprobe) const;

        // Number of clusters (builds the index if needed).
        std::size_t clusterCount() const;

        // Same as searchExact.
        std::vector<SearchResult> search(const std::vector<double> &query, std::size_t k) const;
        std::vector<VectorRecord> listRecords() const;
    };

}
