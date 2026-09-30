#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vectordb/collection.hpp"

namespace vectordb
{

    /**
     * Inverted-file index for cosine similarity search.
     *
     * Build clusters the collection with spherical k-means, then stores an
     * inverted list of member indices per centroid. A query only scans the
     * members of the nprobe most promising clusters instead of every vector.
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

        /**
         * Cluster the records. nlist = 0 selects ceil(sqrt(n)).
         * The seed makes the clustering reproducible across builds.
         * Zero-length vectors have no direction and are excluded from the index.
         */
        void build(const std::unordered_map<std::string, VectorRecord> &records,
                   std::size_t nlist = 0,
                   unsigned seed = 42);

        /**
         * Top-k by cosine similarity, scanning only the nprobe best clusters.
         * nprobe is clamped to [1, clusterCount()]; passing clusterCount()
         * scans everything and is equivalent to an exact search.
         */
        std::vector<SearchResult> search(const std::vector<double> &query,
                                         std::size_t k,
                                         std::size_t nprobe) const;

        std::size_t size() const;
        std::size_t clusterCount() const;
        std::size_t dimension() const;

        /// Member indices of a cluster, indexing into the flat vector arrays.
        const std::vector<std::size_t> &clusterMembers(std::size_t cluster) const;

        /// Unit-norm centroid of a cluster.
        const std::vector<double> &centroid(std::size_t cluster) const;

    private:
        struct Cluster
        {
            std::vector<double> centroid;
            std::vector<std::size_t> members;
        };

        /// (score, cluster) for every cluster, sorted by score descending.
        std::vector<std::pair<double, std::size_t>> rankClusters(const std::vector<double> &query) const;

        void seedCentroids(std::size_t nlist, unsigned seed);
        bool assignStep();
        void updateStep();
        void materializeLists();

        std::vector<std::string> ids_;
        std::vector<std::vector<double>> vectors_; // normalized copies of the records
        std::vector<std::size_t> assignment_;      // assignment_[i] = cluster of vectors_[i]
        std::vector<Cluster> clusters_;
        std::size_t dimension_ = 0;
    };

}
