#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "vectordb/collection.hpp"

namespace vectordb
{

    // Why a search stopped.
    // Certified and Exhausted both mean the answer is exactly right.
    enum class SearchStatus
    {
        Certified,   // the bound proved the answer before all clusters were scanned
        Exhausted,   // every cluster was scanned
        DeadlineHit, // ran out of time (best effort)
        BudgetHit,   // hit the cluster limit (best effort)
    };

    // How much work one search did.
    // A cluster cut off halfway by the deadline is not counted in clustersScanned,
    // but its vectors that were scored are counted in vectorsScanned.
    struct SearchStats
    {
        SearchStatus status = SearchStatus::Exhausted;
        std::size_t clustersScanned = 0;
        std::size_t totalClusters = 0;
        std::size_t vectorsScanned = 0;
        std::size_t totalVectors = 0;
        double elapsedMs = 0.0;
    };

    struct SearchResponse
    {
        std::vector<SearchResult> results;
        SearchStats stats;
    };

    // Limits for one search.
    //   deadline  - time allowed for the search (0 = no limit)
    //   maxProbes - most clusters to scan (0 = no limit)
    //   minProbes - lowest allowed cluster limit; ambiguous queries may raise it
    //               further. 0 turns this off, so maxProbes is used exactly.
    struct SearchParams
    {
        std::chrono::microseconds deadline{0};
        std::size_t minProbes = 1;
        std::size_t maxProbes = 0;
    };

    // Search modes, from fastest to most accurate.
    // Exact is always exactly right. The others may stop early and say so in the status.
    enum class SearchMode
    {
        LowLatency, // 1 ms, at most 1/16 of the clusters
        Balanced,   // 10 ms, at most 1/4 of the clusters
        HighRecall, // 100 ms, no cluster limit
        Exact,      // no limits
    };

    // Returns the limits for a mode, for a collection of n vectors.
    SearchParams paramsForMode(SearchMode mode, std::size_t n);

    // IVF index for cosine similarity search.
    //
    // build() groups the vectors into clusters with k-means. For every cluster
    // we store its centroid, the list of its members, and its radius (the angle
    // from the centroid to the farthest member).
    //
    // From the radius we get an upper bound: the best score any member of the
    // cluster could have for a query. Search visits clusters from the highest
    // bound down, and stops when the k-th best score found so far beats the
    // bound of the next cluster. At that point the answer is proven exact.
    //
    // The index stores normalized (unit length) copies of the vectors, so
    // cosine similarity is just a dot product.
    class IVFIndex
    {
    public:
        // Most k-means rounds to run.
        static constexpr std::size_t kMaxIterations = 20;

        // Small safety margins against rounding errors. They make the bound a
        // little too high rather than too low, so we may miss an early stop but
        // never wrongly claim an answer is exact.
        static constexpr double kRadiusSlack = 1e-9;
        static constexpr double kBoundEpsilon = 1e-9;

        // Inside a cluster, check the clock once every this many vectors.
        static constexpr std::size_t kDeadlineCheckStride = 256;

        // Builds the index. nlist = number of clusters (0 means ceil(sqrt(n))).
        // The same seed always gives the same clusters.
        // Zero vectors have no direction, so they are left out of the index.
        void build(const std::unordered_map<std::string, VectorRecord> &records,
                   std::size_t nlist = 0,
                   unsigned seed = 42);

        // Adds one vector to its nearest cluster without re-running k-means,
        // and widens that cluster's radius if needed. Replaces an old vector
        // with the same id. Returns false if it cannot be added (no clusters
        // yet, or wrong length). A zero vector is accepted but not stored.
        bool add(const std::string &id, const std::vector<double> &values);

        // Removes a vector. Returns false if the id is not in the index.
        // The radius is not shrunk: it is a bit too wide but still correct.
        bool remove(const std::string &id);

        // Exact top-k (same as SearchMode::Exact).
        SearchResponse search(const std::vector<double> &query, std::size_t k) const;

        // Top-k using a mode's limits.
        SearchResponse search(const std::vector<double> &query, std::size_t k, SearchMode mode) const;

        // Top-k using your own limits.
        SearchResponse search(const std::vector<double> &query, std::size_t k, const SearchParams &params) const;

        // Top-k scanning at most nprobe clusters. May still stop earlier if
        // the bound proves the answer.
        std::vector<SearchResult> search(const std::vector<double> &query,
                                         std::size_t k,
                                         std::size_t nprobe) const;

        // Best possible score any member of this cluster can have for the query.
        double upperBound(const std::vector<double> &query, std::size_t cluster) const;

        std::size_t size() const;
        std::size_t clusterCount() const;
        std::size_t dimension() const;

        // Positions (into the index's vector list) of a cluster's members.
        const std::vector<std::size_t> &clusterMembers(std::size_t cluster) const;
        const std::vector<double> &centroid(std::size_t cluster) const;

        // The id of the vector at a position.
        const std::string &id(std::size_t index) const;

    private:
        struct Cluster
        {
            std::vector<double> centroid;
            std::vector<std::size_t> members; // positions in vectors_

            // The radius, stored as its cosine and sine.
            double cosRadius = 1.0;
            double sinRadius = 0.0;
        };

        // One cluster's bound for the current query.
        struct ClusterBound
        {
            double bound;       // best possible score in this cluster
            double centroidDot; // query-centroid score, used to break ties
            std::size_t cluster;
        };

        // If the top two bounds are closer than this, the query is ambiguous.
        static constexpr double kAmbiguityGap = 0.05;

        // Most clusters an ambiguous query can add to the limit.
        static constexpr std::size_t kMaxProbeFloor = 8;

        // k-means steps
        void seedCentroids(std::size_t nlist, unsigned seed);
        bool assignStep();
        void updateStep();
        void buildMemberLists();
        void computeRadii();

        // search helpers
        std::size_t nearestCluster(const std::vector<double> &vec) const;
        std::vector<double> normalizedQuery(const std::vector<double> &query) const;
        double upperBound(double queryCentroidDot, const Cluster &cluster) const;
        static bool scanBefore(const ClusterBound &a, const ClusterBound &b);
        std::vector<ClusterBound> scanOrder(const std::vector<double> &query) const;
        std::size_t adaptiveProbeFloor(const std::vector<ClusterBound> &order) const;
        std::size_t clusterLimit(const SearchParams &params, const std::vector<ClusterBound> &order) const;
        SearchResponse scan(const std::vector<double> &query, std::size_t k, const SearchParams &params) const;

        // Position i in these lists is one vector.
        std::vector<std::string> ids_;
        std::vector<std::vector<double>> vectors_; // normalized copies
        std::vector<std::size_t> assignment_;      // which cluster each vector is in

        std::vector<Cluster> clusters_;
        std::unordered_map<std::string, std::size_t> position_; // id -> position
        std::size_t dimension_ = 0;
    };

}
