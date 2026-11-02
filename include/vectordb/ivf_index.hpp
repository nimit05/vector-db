#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "vectordb/collection.hpp"

namespace vectordb
{

    /**
     * How a search ended. Certified and Exhausted both mean the results are the
     * exact top-k; they differ only in how that was established.
     */
    enum class SearchStatus
    {
        Certified,   // the bound proved the answer exact before every cluster was scanned
        Exhausted,   // every cluster was scanned, so the answer is trivially exact
        DeadlineHit, // stopped at the latency deadline; best effort
        BudgetHit,   // stopped at the probe cap; best effort
    };

    /**
     * How much of the index a search got through. clustersScanned counts only
     * clusters scanned to the end: a search cut off by its deadline partway
     * through a cluster reports that cluster's vectors in vectorsScanned but
     * not the cluster itself, so the two never claim more coverage than there was.
     */
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

    /**
     * The latency contract of one search.
     *
     * deadline is a budget measured from the moment the search starts, not a
     * point in time; 0 means no deadline. It is checked at every cluster
     * boundary and periodically inside a cluster, so one oversized inverted
     * list cannot overrun it. A search that runs out of time returns its
     * best-so-far, never an empty result.
     *
     * maxProbes caps how many clusters are scanned; 0 means no cap. minProbes
     * raises that cap, and is itself raised for a query the bound cannot
     * resolve (see the adaptive floor in ivf_index.cpp). minProbes = 0 disables
     * both floors, so maxProbes is then honoured exactly.
     *
     * Neither floor nor cap delays a certificate: an answer the bound has
     * proven exact is returned as soon as it is proven.
     */
    struct SearchParams
    {
        std::chrono::microseconds deadline{0};
        std::size_t minProbes = 1;
        std::size_t maxProbes = 0;
    };

    /**
     * The entire user-facing tuning surface. The modes split into two
     * contracts:
     *
     *   Exact                    guaranteed exact. Terminates as Certified when
     *                            the bound fires and Exhausted otherwise, never
     *                            as DeadlineHit or BudgetHit.
     *
     *   LowLatency / Balanced /  best effort. Certified or Exhausted when the
     *   HighRecall               data makes it free, DeadlineHit or BudgetHit
     *                            otherwise.
     *
     * Exact is therefore a guarantee about the answer, not merely an absence of
     * limits: the early exit is an optimization within it, never its definition.
     */
    enum class SearchMode
    {
        LowLatency,
        Balanced,
        HighRecall,
        Exact,
    };

    /// Preset parameters for a mode over a collection of n vectors. Probe caps
    /// are fractions of the default cluster count, so a mode means the same
    /// share of the index whatever the collection size.
    SearchParams paramsForMode(SearchMode mode, std::size_t n);

    /**
     * Inverted-file index for cosine similarity search.
     *
     * Build clusters the collection with spherical k-means, then stores an
     * inverted list of member indices per centroid, plus the angular radius of
     * each cluster: the widest angle between its centroid and any member.
     *
     * The radius gives an upper bound on how similar any member of a cluster
     * can be to a query. Search scans clusters best-bound-first and stops once
     * the k-th best score found beats the bound of every unscanned cluster, at
     * which point the answer is provably exact.
     *
     * The index keeps its own L2-normalized copy of the data, so cosine
     * similarity reduces to a plain dot product. Records in the Collection are
     * left untouched.
     */
    class IVFIndex
    {
    public:
        /// Default cap on k-means iterations before the build gives up.
        static constexpr std::size_t kMaxIterations = 20;

        /// Subtracted from each cluster's cosine radius at build. Widening every
        /// cone keeps the bound an over-estimate despite rounding error, so
        /// numerical error can only cost an early stop, never a false certificate.
        static constexpr double kRadiusSlack = 1e-9;

        /// Margin the k-th best score must clear above the next cluster's bound
        /// before the search stops. It sits on the bound's side for the same reason.
        static constexpr double kBoundEpsilon = 1e-9;

        /// How many vectors a search scans between two clock reads inside one
        /// cluster. Large enough to keep the read off the hot path, small enough
        /// that a single huge inverted list cannot run past the deadline.
        static constexpr std::size_t kDeadlineCheckStride = 256;

        /**
         * Cluster the records. nlist = 0 selects ceil(sqrt(n)).
         * Centroids are seeded with k-means++; the seed makes the clustering
         * reproducible across builds.
         * Zero-length vectors have no direction and are excluded from the index.
         * This deliberately differs from Collection::searchExact, which throws
         * when the collection holds a zero record: the same collection can make
         * searchExact throw while IVF search succeeds.
         */
        void build(const std::unordered_map<std::string, VectorRecord> &records,
                   std::size_t nlist = 0,
                   unsigned seed = 42);

        /**
         * Exact top-k by cosine similarity. Scans clusters in decreasing bound
         * order and stops as soon as the result is provably exact. The status is
         * Certified if it stopped early, Exhausted if it scanned every cluster.
         */
        SearchResponse search(const std::vector<double> &query, std::size_t k) const;

        /**
         * Top-k under a mode's latency contract. Exact is guaranteed exact;
         * the other modes are best effort and say so in the status.
         */
        SearchResponse search(const std::vector<double> &query, std::size_t k, SearchMode mode) const;

        /// Top-k under explicit parameters, for callers that need a deadline or
        /// a probe cap of their own rather than one of the presets.
        SearchResponse search(const std::vector<double> &query, std::size_t k, const SearchParams &params) const;

        /**
         * Top-k scanning at most nprobe clusters, in the same bound order as the
         * exact search, and stopping sooner if the answer is certified first.
         * nprobe is clamped to [1, clusterCount()]; passing clusterCount()
         * scans everything and is equivalent to an exact search.
         */
        std::vector<SearchResult> search(const std::vector<double> &query,
                                         std::size_t k,
                                         std::size_t nprobe) const;

        /**
         * Upper bound on the cosine similarity between the query and any member
         * of the cluster. Never below the true maximum.
         */
        double upperBound(const std::vector<double> &query, std::size_t cluster) const;

        std::size_t size() const;
        std::size_t clusterCount() const;
        std::size_t dimension() const;

        /// Member indices of a cluster, indexing into the flat vector arrays.
        const std::vector<std::size_t> &clusterMembers(std::size_t cluster) const;

        /// Unit-norm centroid of a cluster.
        const std::vector<double> &centroid(std::size_t cluster) const;

        /// Record id of the vector at a flat index, as returned by clusterMembers.
        const std::string &id(std::size_t index) const;

    private:
        struct Cluster
        {
            std::vector<double> centroid;
            std::vector<std::size_t> members;

            // Cosine and sine of the widest angle between the centroid and any
            // member, widened by kRadiusSlack. Every member lies inside this cone.
            double cosRadius = 1.0;
            double sinRadius = 0.0;
        };

        struct ClusterBound
        {
            double bound;       // upper bound on any member's similarity to the query
            double centroidDot; // query-centroid similarity, used to break ties
            std::size_t cluster;
        };

        /// Validates the query against the index and returns a unit-length copy.
        std::vector<double> normalizedQuery(const std::vector<double> &query) const;

        /// The bound itself, from the query-centroid dot product the caller already has.
        double upperBound(double queryCentroidDot, const Cluster &cluster) const;

        /// Every cluster, ordered by bound descending: the order search scans them in.
        std::vector<ClusterBound> scanOrder(const std::vector<double> &query) const;

        /// Lower limit on probes for this query, from the gap between the top
        /// two bounds. Covers the case where the bound is too loose to fire.
        std::size_t adaptiveProbeFloor(const std::vector<ClusterBound> &order) const;

        /// Shared search loop, under one latency contract.
        SearchResponse scan(const std::vector<double> &query, std::size_t k, const SearchParams &params) const;

        /// Bound gap above which the leading cluster counts as unambiguous.
        static constexpr double kAmbiguityGap = 0.05;

        /// Ceiling on the adaptive floor, so a query that straddles every cone
        /// cannot turn a low-latency search into a full scan.
        static constexpr std::size_t kMaxProbeFloor = 8;

        void seedCentroids(std::size_t nlist, unsigned seed);
        bool assignStep();
        void updateStep();
        void materializeLists();
        void computeRadii();

        std::vector<std::string> ids_;
        std::vector<std::vector<double>> vectors_; // normalized copies of the records
        std::vector<std::size_t> assignment_;      // assignment_[i] = cluster of vectors_[i]
        std::vector<Cluster> clusters_;
        std::size_t dimension_ = 0;
    };

}
