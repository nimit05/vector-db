#include "vectordb/ivf_index.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>

namespace vectordb
{

    namespace
    {
        constexpr double kNormEpsilon = 1e-12;
        constexpr std::size_t kUnassigned = std::numeric_limits<std::size_t>::max();

        double dot(const std::vector<double> &a, const std::vector<double> &b)
        {
            double sum = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i)
            {
                sum += a[i] * b[i];
            }
            return sum;
        }

        /// Scales the vector to unit length. Returns false for a zero vector,
        /// which has no direction and cannot be normalized.
        bool normalize(std::vector<double> &values)
        {
            double norm = 0.0;
            for (double value : values)
            {
                norm += value * value;
            }
            norm = std::sqrt(norm);

            if (norm < kNormEpsilon)
            {
                return false;
            }

            for (double &value : values)
            {
                value /= norm;
            }
            return true;
        }
    }

    void IVFIndex::build(const std::unordered_map<std::string, VectorRecord> &records,
                         std::size_t nlist,
                         unsigned seed)
    {
        ids_.clear();
        vectors_.clear();
        assignment_.clear();
        clusters_.clear();
        dimension_ = 0;

        if (records.empty())
        {
            return;
        }

        // unordered_map iteration order is unspecified, so walk the records in
        // sorted id order instead. Without this a fixed seed would still give a
        // different clustering from run to run.
        std::vector<const VectorRecord *> ordered;
        ordered.reserve(records.size());
        for (const auto &[id, record] : records)
        {
            ordered.push_back(&record);
        }
        std::sort(ordered.begin(), ordered.end(),
                  [](const VectorRecord *a, const VectorRecord *b)
                  {
                      return a->getId() < b->getId();
                  });

        dimension_ = ordered.front()->dimension();

        for (const VectorRecord *record : ordered)
        {
            if (record->dimension() != dimension_)
            {
                throw std::invalid_argument("Collection contains vectors of differing dimensions");
            }

            std::vector<double> values = record->getValues();
            if (!normalize(values))
            {
                continue; // zero vector: no direction to cluster on
            }

            ids_.push_back(record->getId());
            vectors_.push_back(std::move(values));
        }

        const std::size_t n = vectors_.size();
        if (n == 0)
        {
            return;
        }

        if (nlist == 0)
        {
            nlist = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n))));
        }
        nlist = std::clamp<std::size_t>(nlist, 1, n);

        seedCentroids(nlist, seed);
        assignment_.assign(n, kUnassigned);

        for (std::size_t iteration = 0; iteration < kMaxIterations; ++iteration)
        {
            if (!assignStep())
            {
                break; // converged: no vector changed cluster
            }
            updateStep();
        }

        materializeLists();
        computeRadii();
    }

    void IVFIndex::seedCentroids(std::size_t nlist, unsigned seed)
    {
        std::vector<std::size_t> indices(vectors_.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937 generator(seed);
        std::shuffle(indices.begin(), indices.end(), generator);

        clusters_.resize(nlist);
        for (std::size_t c = 0; c < nlist; ++c)
        {
            clusters_[c].centroid = vectors_[indices[c]];
        }
    }

    bool IVFIndex::assignStep()
    {
        bool changed = false;

        for (std::size_t i = 0; i < vectors_.size(); ++i)
        {
            double bestScore = -std::numeric_limits<double>::infinity();
            std::size_t bestCluster = 0;

            for (std::size_t c = 0; c < clusters_.size(); ++c)
            {
                const double score = dot(vectors_[i], clusters_[c].centroid);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestCluster = c;
                }
            }

            if (assignment_[i] != bestCluster)
            {
                assignment_[i] = bestCluster;
                changed = true;
            }
        }

        return changed;
    }

    void IVFIndex::updateStep()
    {
        const std::size_t nlist = clusters_.size();

        std::vector<std::vector<double>> sums(nlist, std::vector<double>(dimension_, 0.0));
        std::vector<std::size_t> counts(nlist, 0);

        for (std::size_t i = 0; i < vectors_.size(); ++i)
        {
            const std::size_t c = assignment_[i];
            ++counts[c];
            for (std::size_t d = 0; d < dimension_; ++d)
            {
                sums[c][d] += vectors_[i][d];
            }
        }

        for (std::size_t c = 0; c < nlist; ++c)
        {
            if (counts[c] == 0)
            {
                continue; // handled below
            }

            // Spherical k-means: the centroid is a direction, so the mean is
            // pushed back onto the unit sphere. A mean that cancels out to zero
            // keeps its previous centroid.
            std::vector<double> mean = sums[c];
            if (normalize(mean))
            {
                clusters_[c].centroid = std::move(mean);
            }
        }

        // An empty cluster has no centroid to speak of and would silently leave
        // us with fewer than nlist clusters. Reseed it to the point currently
        // worst served by its own centroid, taking care not to empty the donor.
        for (std::size_t c = 0; c < nlist; ++c)
        {
            if (counts[c] > 0)
            {
                continue;
            }

            double worstScore = std::numeric_limits<double>::infinity();
            std::size_t worstIndex = kUnassigned;

            for (std::size_t i = 0; i < vectors_.size(); ++i)
            {
                const std::size_t owner = assignment_[i];
                if (counts[owner] < 2)
                {
                    continue;
                }

                const double score = dot(vectors_[i], clusters_[owner].centroid);
                if (score < worstScore)
                {
                    worstScore = score;
                    worstIndex = i;
                }
            }

            if (worstIndex == kUnassigned)
            {
                continue; // every cluster is a singleton; nothing to donate
            }

            --counts[assignment_[worstIndex]];
            assignment_[worstIndex] = c;
            counts[c] = 1;
            clusters_[c].centroid = vectors_[worstIndex];
        }
    }

    void IVFIndex::materializeLists()
    {
        for (Cluster &cluster : clusters_)
        {
            cluster.members.clear();
        }

        for (std::size_t i = 0; i < vectors_.size(); ++i)
        {
            clusters_[assignment_[i]].members.push_back(i);
        }
    }

    void IVFIndex::computeRadii()
    {
        for (Cluster &cluster : clusters_)
        {
            // The widest member angle is the one with the smallest dot product.
            // Starting at 1.0 rather than the first member keeps an empty cluster
            // (which no bound can get wrong) well defined.
            double minDot = 1.0;
            for (std::size_t index : cluster.members)
            {
                minDot = std::min(minDot, dot(cluster.centroid, vectors_[index]));
            }

            // Inflate once, here, and derive the sine from the inflated cosine so
            // the two describe the same widened cone.
            cluster.cosRadius = std::clamp(minDot - kRadiusSlack, -1.0, 1.0);
            cluster.sinRadius = std::sqrt(std::max(0.0, 1.0 - cluster.cosRadius * cluster.cosRadius));
        }
    }

    std::vector<double> IVFIndex::normalizedQuery(const std::vector<double> &query) const
    {
        if (query.size() != dimension_)
        {
            throw std::invalid_argument("Query vector dimension does not match collection vectors");
        }

        std::vector<double> normalized = query;
        if (!normalize(normalized))
        {
            throw std::invalid_argument("Zero vector is not allowed");
        }
        return normalized;
    }

    double IVFIndex::upperBound(double queryCentroidDot, const Cluster &cluster) const
    {
        // With θq the query-centroid angle and R the cluster radius, no member is
        // closer to the query than θq − R, so no member scores above cos(θq − R).
        // Expanding the angle difference keeps this in cosine space: acos is
        // ill-conditioned as the dot product approaches 1, which is exactly the
        // tight-cluster case.
        const double cq = std::clamp(queryCentroidDot, -1.0, 1.0);

        if (cq >= cluster.cosRadius)
        {
            return 1.0; // the query sits inside the cluster's cone
        }

        return cq * cluster.cosRadius + std::sqrt(std::max(0.0, 1.0 - cq * cq)) * cluster.sinRadius;
    }

    double IVFIndex::upperBound(const std::vector<double> &query, std::size_t cluster) const
    {
        const Cluster &target = clusters_.at(cluster);
        return upperBound(dot(normalizedQuery(query), target.centroid), target);
    }

    std::vector<IVFIndex::ClusterBound> IVFIndex::scanOrder(const std::vector<double> &query) const
    {
        std::vector<ClusterBound> order;
        order.reserve(clusters_.size());

        for (std::size_t c = 0; c < clusters_.size(); ++c)
        {
            const double centroidDot = dot(query, clusters_[c].centroid);
            order.push_back({upperBound(centroidDot, clusters_[c]), centroidDot, c});
        }

        // Bound descending is what the stopping rule needs: every unscanned
        // cluster is then bounded by the next one in line. Among equal bounds,
        // typically several clusters whose cone contains the query, the nearest
        // centroid goes first; the cluster index makes the order deterministic.
        std::sort(order.begin(), order.end(),
                  [](const ClusterBound &a, const ClusterBound &b)
                  {
                      if (a.bound != b.bound)
                      {
                          return a.bound > b.bound;
                      }
                      if (a.centroidDot != b.centroidDot)
                      {
                          return a.centroidDot > b.centroidDot;
                      }
                      return a.cluster < b.cluster;
                  });

        return order;
    }

    SearchResponse IVFIndex::search(const std::vector<double> &query, std::size_t k) const
    {
        return scan(query, k, clusters_.size());
    }

    std::vector<SearchResult> IVFIndex::search(const std::vector<double> &query,
                                               std::size_t k,
                                               std::size_t nprobe) const
    {
        return scan(query, k, std::max<std::size_t>(nprobe, 1)).results;
    }

    SearchResponse IVFIndex::scan(const std::vector<double> &query,
                                  std::size_t k,
                                  std::size_t maxProbes) const
    {
        const auto start = std::chrono::steady_clock::now();

        SearchResponse response;
        SearchStats &stats = response.stats;
        stats.totalClusters = clusters_.size();
        stats.totalVectors = vectors_.size();

        const auto finish = [&](SearchStatus status)
        {
            stats.status = status;
            stats.elapsedMs = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
            return std::move(response);
        };

        if (vectors_.empty())
        {
            return finish(SearchStatus::Exhausted);
        }
        if (k == 0)
        {
            return finish(SearchStatus::Certified); // the empty answer needs no scan
        }

        const std::vector<double> normalized = normalizedQuery(query);
        const std::vector<ClusterBound> order = scanOrder(normalized);
        const std::size_t probeLimit = std::min(maxProbes, order.size());

        struct Candidate
        {
            double score;
            std::size_t index;
        };

        // Ranks a candidate against another: higher score wins, ties break on
        // the smaller id so results stay deterministic. Used as the comparator
        // of a priority_queue, which puts the *lowest*-ranked candidate on top —
        // exactly the one to evict once the heap holds more than k.
        const auto better = [this](const Candidate &a, const Candidate &b)
        {
            if (a.score != b.score)
            {
                return a.score > b.score;
            }
            return ids_[a.index] < ids_[b.index];
        };

        std::priority_queue<Candidate, std::vector<Candidate>, decltype(better)> heap(better);

        SearchStatus status = SearchStatus::Exhausted;

        for (std::size_t next = 0; next < order.size();)
        {
            if (next == probeLimit)
            {
                status = SearchStatus::BudgetHit;
                break;
            }

            const Cluster &cluster = clusters_[order[next].cluster];
            for (std::size_t index : cluster.members)
            {
                heap.push({dot(normalized, vectors_[index]), index});
                if (heap.size() > k)
                {
                    heap.pop();
                }
            }

            ++next;
            ++stats.clustersScanned;
            stats.vectorsScanned += cluster.members.size();

            // Every unscanned cluster is bounded by order[next].bound. Once the
            // k-th best score strictly clears that bound, no unscanned vector can
            // match it, let alone outrank it on the id tie-break. The check needs
            // a full heap: with fewer than k candidates there is no k-th score.
            if (next < order.size() && heap.size() == k &&
                heap.top().score > order[next].bound + kBoundEpsilon)
            {
                status = SearchStatus::Certified;
                break;
            }
        }

        // The heap drains worst-first, so fill back to front.
        response.results.resize(heap.size());
        for (std::size_t i = heap.size(); i-- > 0;)
        {
            response.results[i] = {ids_[heap.top().index], heap.top().score};
            heap.pop();
        }

        return finish(status);
    }

    std::size_t IVFIndex::size() const
    {
        return vectors_.size();
    }

    std::size_t IVFIndex::clusterCount() const
    {
        return clusters_.size();
    }

    std::size_t IVFIndex::dimension() const
    {
        return dimension_;
    }

    const std::vector<std::size_t> &IVFIndex::clusterMembers(std::size_t cluster) const
    {
        return clusters_.at(cluster).members;
    }

    const std::vector<double> &IVFIndex::centroid(std::size_t cluster) const
    {
        return clusters_.at(cluster).centroid;
    }

    const std::string &IVFIndex::id(std::size_t index) const
    {
        return ids_.at(index);
    }

}
