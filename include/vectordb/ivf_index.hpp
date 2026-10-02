#pragma once

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

        /**
         * Cluster the records. nlist = 0 selects ceil(sqrt(n)).
         * The seed makes the clustering reproducible across builds.
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

        /// Shared search loop. Scans at most maxProbes clusters.
        SearchResponse scan(const std::vector<double> &query, std::size_t k, std::size_t maxProbes) const;

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
