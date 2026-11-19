#include "vectordb/ivf_index.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>

namespace vectordb
{

    using Clock = std::chrono::steady_clock;

    namespace
    {
        const std::size_t kNone = std::numeric_limits<std::size_t>::max();

        double dot(const std::vector<double> &a, const std::vector<double> &b)
        {
            double sum = 0.0;
            for (std::size_t i = 0; i < a.size(); i++)
            {
                sum += a[i] * b[i];
            }
            return sum;
        }

        // Makes the vector length 1. Returns false for a zero vector,
        // because it has no direction.
        bool normalize(std::vector<double> &values)
        {
            double length = 0.0;
            for (double v : values)
            {
                length += v * v;
            }
            length = std::sqrt(length);

            if (length < 1e-12)
            {
                return false;
            }

            for (double &v : values)
            {
                v /= length;
            }
            return true;
        }

        // Distance between two unit vectors: 0 when equal, 2 when opposite.
        double distance(const std::vector<double> &a, const std::vector<double> &b)
        {
            return std::max(0.0, 1.0 - dot(a, b));
        }

        double millisecondsSince(Clock::time_point start)
        {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }

        bool timeIsUp(Clock::time_point start, std::chrono::microseconds deadline)
        {
            if (deadline.count() == 0)
            {
                return false; // no deadline
            }
            return Clock::now() - start >= deadline;
        }

        // A possible result while searching.
        struct Candidate
        {
            double score;
            std::string id;
        };

        // Higher score wins. If the scores are equal, the smaller id wins,
        // so results always come out in the same order.
        bool ranksHigher(const Candidate &a, const Candidate &b)
        {
            if (a.score != b.score)
            {
                return a.score > b.score;
            }
            return a.id < b.id;
        }

        // std::priority_queue keeps the "biggest" item on top. We want the
        // WORST candidate on top, so we can throw it out quickly when a better
        // one comes along. So we tell it that better candidates are "smaller".
        struct WorstOnTop
        {
            bool operator()(const Candidate &a, const Candidate &b) const
            {
                return ranksHigher(a, b);
            }
        };
    }

    // ------------------------------------------------------------------
    // Search modes
    // ------------------------------------------------------------------

    SearchParams paramsForMode(SearchMode mode, std::size_t n)
    {
        // Default number of clusters is ceil(sqrt(n)).
        std::size_t numClusters = 1;
        if (n > 0)
        {
            numClusters = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n))));
        }

        SearchParams params;

        if (mode == SearchMode::LowLatency)
        {
            params.deadline = std::chrono::milliseconds(1);
            params.maxProbes = std::max<std::size_t>(1, numClusters / 16);
        }
        else if (mode == SearchMode::Balanced)
        {
            params.deadline = std::chrono::milliseconds(10);
            params.maxProbes = std::max<std::size_t>(1, numClusters / 4);
        }
        else if (mode == SearchMode::HighRecall)
        {
            params.deadline = std::chrono::milliseconds(100); // no cluster limit
        }
        // Exact: no limits at all.

        return params;
    }

    // ------------------------------------------------------------------
    // Building the index (k-means)
    // ------------------------------------------------------------------

    void IVFIndex::build(const std::unordered_map<std::string, VectorRecord> &records,
                         std::size_t nlist,
                         unsigned seed)
    {
        ids_.clear();
        vectors_.clear();
        assignment_.clear();
        clusters_.clear();
        position_.clear();
        dimension_ = 0;

        if (records.empty())
        {
            return;
        }

        // An unordered_map has no fixed order. Sort the ids so the same data
        // always gives the same clusters.
        std::vector<std::string> sortedIds;
        for (const auto &entry : records)
        {
            sortedIds.push_back(entry.first);
        }
        std::sort(sortedIds.begin(), sortedIds.end());

        dimension_ = records.at(sortedIds[0]).dimension();

        // Store a normalized copy of every vector.
        for (const std::string &id : sortedIds)
        {
            const VectorRecord &record = records.at(id);
            if (record.dimension() != dimension_)
            {
                throw std::invalid_argument("Collection contains vectors of differing dimensions");
            }

            std::vector<double> values = record.getValues();
            if (!normalize(values))
            {
                continue; // skip zero vectors
            }

            position_[id] = ids_.size();
            ids_.push_back(id);
            vectors_.push_back(values);
        }

        std::size_t n = vectors_.size();
        if (n == 0)
        {
            return;
        }

        if (nlist == 0)
        {
            nlist = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n))));
        }
        nlist = std::clamp<std::size_t>(nlist, 1, n);

        // k-means: pick starting centroids, then repeat
        // "assign each vector to its nearest centroid" and "move each centroid
        // to the middle of its vectors" until nothing changes.
        seedCentroids(nlist, seed);
        assignment_.assign(n, kNone);

        for (std::size_t round = 0; round < kMaxIterations; round++)
        {
            bool changed = assignStep();
            if (!changed)
            {
                break;
            }
            updateStep();
        }

        buildMemberLists();
        computeRadii();
    }

    // k-means++: the first centroid is a random vector. Each next one is also
    // random, but vectors far from the centroids picked so far are more likely
    // to be chosen. This spreads the starting centroids out.
    void IVFIndex::seedCentroids(std::size_t nlist, unsigned seed)
    {
        std::size_t n = vectors_.size();
        std::mt19937 random(seed);
        std::uniform_int_distribution<std::size_t> randomVector(0, n - 1);

        clusters_.assign(nlist, Cluster{});
        clusters_[0].centroid = vectors_[randomVector(random)];

        // nearest[i] = distance from vector i to its closest chosen centroid
        std::vector<double> nearest(n);
        for (std::size_t i = 0; i < n; i++)
        {
            nearest[i] = distance(vectors_[i], clusters_[0].centroid);
        }

        for (std::size_t c = 1; c < nlist; c++)
        {
            // Each vector's chance of being picked is distance squared.
            // runningTotal[i] = sum of the chances of vectors 0..i
            std::vector<double> runningTotal(n);
            double total = 0.0;
            for (std::size_t i = 0; i < n; i++)
            {
                total += nearest[i] * nearest[i];
                runningTotal[i] = total;
            }

            std::size_t chosen = 0;
            if (total == 0.0)
            {
                // Every vector already sits on a centroid, so any pick is as good.
                chosen = randomVector(random);
            }
            else
            {
                // Pick a random point in [0, total) and find whose range it falls in.
                double target = std::uniform_real_distribution<double>(0.0, total)(random);
                chosen = kNone;
                for (std::size_t i = 0; i < n; i++)
                {
                    if (runningTotal[i] > target)
                    {
                        chosen = i;
                        break;
                    }
                }
                if (chosen == kNone)
                {
                    // Rounding put target at the very end: take the last vector with a chance.
                    for (std::size_t i = 0; i < n; i++)
                    {
                        if (runningTotal[i] >= total)
                        {
                            chosen = i;
                            break;
                        }
                    }
                }
            }

            clusters_[c].centroid = vectors_[chosen];

            for (std::size_t i = 0; i < n; i++)
            {
                nearest[i] = std::min(nearest[i], distance(vectors_[i], clusters_[c].centroid));
            }
        }
    }

    // Index of the centroid with the highest dot product with vec.
    std::size_t IVFIndex::nearestCluster(const std::vector<double> &vec) const
    {
        std::size_t best = 0;
        double bestScore = -std::numeric_limits<double>::infinity();
        for (std::size_t c = 0; c < clusters_.size(); c++)
        {
            double score = dot(vec, clusters_[c].centroid);
            if (score > bestScore)
            {
                bestScore = score;
                best = c;
            }
        }
        return best;
    }

    // Puts every vector in its nearest cluster. Returns true if any vector moved.
    bool IVFIndex::assignStep()
    {
        bool changed = false;
        for (std::size_t i = 0; i < vectors_.size(); i++)
        {
            std::size_t best = nearestCluster(vectors_[i]);
            if (assignment_[i] != best)
            {
                assignment_[i] = best;
                changed = true;
            }
        }
        return changed;
    }

    // Moves every centroid to the average direction of its vectors.
    void IVFIndex::updateStep()
    {
        std::size_t nlist = clusters_.size();
        std::vector<std::vector<double>> sums(nlist, std::vector<double>(dimension_, 0.0));
        std::vector<std::size_t> counts(nlist, 0);

        for (std::size_t i = 0; i < vectors_.size(); i++)
        {
            std::size_t c = assignment_[i];
            counts[c]++;
            for (std::size_t d = 0; d < dimension_; d++)
            {
                sums[c][d] += vectors_[i][d];
            }
        }

        for (std::size_t c = 0; c < nlist; c++)
        {
            if (counts[c] == 0)
            {
                continue; // empty clusters are fixed below
            }
            // Normalize the sum to get the average direction. If the vectors
            // cancel out to zero, keep the old centroid.
            std::vector<double> mean = sums[c];
            if (normalize(mean))
            {
                clusters_[c].centroid = mean;
            }
        }

        // An empty cluster is useless. Give it the vector that is farthest
        // from its own centroid, but never take the last vector of a cluster.
        for (std::size_t c = 0; c < nlist; c++)
        {
            if (counts[c] > 0)
            {
                continue;
            }

            std::size_t worst = kNone;
            double worstScore = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < vectors_.size(); i++)
            {
                std::size_t owner = assignment_[i];
                if (counts[owner] < 2)
                {
                    continue;
                }
                double score = dot(vectors_[i], clusters_[owner].centroid);
                if (score < worstScore)
                {
                    worstScore = score;
                    worst = i;
                }
            }

            if (worst == kNone)
            {
                continue; // every cluster has one vector, nothing to give
            }

            counts[assignment_[worst]]--;
            assignment_[worst] = c;
            counts[c] = 1;
            clusters_[c].centroid = vectors_[worst];
        }
    }

    // Fills each cluster's member list from assignment_.
    void IVFIndex::buildMemberLists()
    {
        for (Cluster &cluster : clusters_)
        {
            cluster.members.clear();
        }
        for (std::size_t i = 0; i < vectors_.size(); i++)
        {
            clusters_[assignment_[i]].members.push_back(i);
        }
    }

    // Radius = angle from the centroid to the farthest member.
    // The farthest member is the one with the smallest dot product.
    void IVFIndex::computeRadii()
    {
        for (Cluster &cluster : clusters_)
        {
            double smallestDot = 1.0;
            for (std::size_t i : cluster.members)
            {
                smallestDot = std::min(smallestDot, dot(cluster.centroid, vectors_[i]));
            }

            // Widen a tiny bit for rounding errors.
            cluster.cosRadius = std::clamp(smallestDot - kRadiusSlack, -1.0, 1.0);
            cluster.sinRadius = std::sqrt(std::max(0.0, 1.0 - cluster.cosRadius * cluster.cosRadius));
        }
    }

    // ------------------------------------------------------------------
    // Adding and removing single vectors
    // ------------------------------------------------------------------

    bool IVFIndex::add(const std::string &id, const std::vector<double> &values)
    {
        if (clusters_.empty() || values.size() != dimension_)
        {
            return false;
        }

        remove(id); // if this id is already here, drop the old vector first

        std::vector<double> vec = values;
        if (!normalize(vec))
        {
            return true; // zero vector: not stored, same as in build()
        }

        std::size_t c = nearestCluster(vec);
        double score = dot(vec, clusters_[c].centroid);

        std::size_t index = vectors_.size();
        vectors_.push_back(vec);
        ids_.push_back(id);
        assignment_.push_back(c);
        position_[id] = index;
        clusters_[c].members.push_back(index);

        // If the new vector is outside the cluster's radius, widen the radius.
        // Otherwise the bound would be too low and search could skip this
        // cluster by mistake.
        double newCos = std::clamp(score - kRadiusSlack, -1.0, 1.0);
        if (newCos < clusters_[c].cosRadius)
        {
            clusters_[c].cosRadius = newCos;
            clusters_[c].sinRadius = std::sqrt(std::max(0.0, 1.0 - newCos * newCos));
        }
        return true;
    }

    bool IVFIndex::remove(const std::string &id)
    {
        if (position_.count(id) == 0)
        {
            return false;
        }

        std::size_t index = position_[id];
        position_.erase(id);

        // Take it out of its cluster's member list.
        std::vector<std::size_t> &members = clusters_[assignment_[index]].members;
        members.erase(std::find(members.begin(), members.end(), index));

        // Move the last vector into the empty spot, so no other positions change.
        std::size_t last = vectors_.size() - 1;
        if (index != last)
        {
            vectors_[index] = vectors_[last];
            ids_[index] = ids_[last];
            assignment_[index] = assignment_[last];
            position_[ids_[index]] = index;

            // Its cluster still lists it at the old position; fix that.
            std::vector<std::size_t> &movedMembers = clusters_[assignment_[index]].members;
            for (std::size_t &member : movedMembers)
            {
                if (member == last)
                {
                    member = index;
                }
            }
        }

        vectors_.pop_back();
        ids_.pop_back();
        assignment_.pop_back();
        return true;
    }

    // ------------------------------------------------------------------
    // Bounds
    // ------------------------------------------------------------------

    std::vector<double> IVFIndex::normalizedQuery(const std::vector<double> &query) const
    {
        if (query.size() != dimension_)
        {
            throw std::invalid_argument("Query vector dimension does not match collection vectors");
        }
        std::vector<double> q = query;
        if (!normalize(q))
        {
            throw std::invalid_argument("Zero vector is not allowed");
        }
        return q;
    }

    // Best possible score in the cluster: cos(angle to centroid - radius).
    // We use cos(a - b) = cos a * cos b + sin a * sin b, so no angles are needed.
    double IVFIndex::upperBound(double queryCentroidDot, const Cluster &cluster) const
    {
        double cosA = std::clamp(queryCentroidDot, -1.0, 1.0);

        if (cosA >= cluster.cosRadius)
        {
            return 1.0; // the query is inside the cluster's radius
        }

        double sinA = std::sqrt(std::max(0.0, 1.0 - cosA * cosA));
        return cosA * cluster.cosRadius + sinA * cluster.sinRadius;
    }

    double IVFIndex::upperBound(const std::vector<double> &query, std::size_t cluster) const
    {
        const Cluster &c = clusters_.at(cluster);
        return upperBound(dot(normalizedQuery(query), c.centroid), c);
    }

    // Used to sort clusters: highest bound first. On equal bounds, the
    // nearer centroid first, then the lower cluster number.
    bool IVFIndex::scanBefore(const ClusterBound &a, const ClusterBound &b)
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
    }

    // All clusters with their bounds, sorted so the highest bound comes first.
    std::vector<IVFIndex::ClusterBound> IVFIndex::scanOrder(const std::vector<double> &query) const
    {
        std::vector<ClusterBound> order;
        for (std::size_t c = 0; c < clusters_.size(); c++)
        {
            double centroidDot = dot(query, clusters_[c].centroid);
            order.push_back({upperBound(centroidDot, clusters_[c]), centroidDot, c});
        }
        std::sort(order.begin(), order.end(), scanBefore);
        return order;
    }

    // ------------------------------------------------------------------
    // Search
    // ------------------------------------------------------------------

    // How many clusters an ambiguous query should be allowed to scan.
    // If the top two bounds are far apart, the query clearly belongs to one
    // cluster: 1 is enough. The closer they are, the more clusters (up to 8).
    std::size_t IVFIndex::adaptiveProbeFloor(const std::vector<ClusterBound> &order) const
    {
        if (order.size() < 2)
        {
            return 1;
        }

        double gap = order[0].bound - order[1].bound;
        if (gap >= kAmbiguityGap)
        {
            return 1;
        }

        // gap 0.05 -> 1 cluster ... gap 0 -> 8 clusters
        double ambiguity = 1.0 - std::max(0.0, gap) / kAmbiguityGap;
        double extra = ambiguity * (kMaxProbeFloor - 1);
        return 1 + static_cast<std::size_t>(std::llround(extra));
    }

    // Most clusters this search may scan.
    std::size_t IVFIndex::clusterLimit(const SearchParams &params,
                                       const std::vector<ClusterBound> &order) const
    {
        if (params.maxProbes == 0)
        {
            return order.size(); // no limit
        }

        std::size_t limit = params.maxProbes;
        if (params.minProbes > 0)
        {
            limit = std::max(limit, params.minProbes);
            limit = std::max(limit, adaptiveProbeFloor(order));
        }
        return std::min(limit, order.size());
    }

    SearchResponse IVFIndex::search(const std::vector<double> &query, std::size_t k) const
    {
        return scan(query, k, paramsForMode(SearchMode::Exact, vectors_.size()));
    }

    SearchResponse IVFIndex::search(const std::vector<double> &query, std::size_t k, SearchMode mode) const
    {
        return scan(query, k, paramsForMode(mode, vectors_.size()));
    }

    SearchResponse IVFIndex::search(const std::vector<double> &query,
                                    std::size_t k,
                                    const SearchParams &params) const
    {
        return scan(query, k, params);
    }

    std::vector<SearchResult> IVFIndex::search(const std::vector<double> &query,
                                               std::size_t k,
                                               std::size_t nprobe) const
    {
        // minProbes = 0: scan at most nprobe clusters, never more.
        SearchParams params;
        params.minProbes = 0;
        params.maxProbes = std::max<std::size_t>(nprobe, 1);
        return scan(query, k, params).results;
    }

    // The search loop that every search() above uses.
    SearchResponse IVFIndex::scan(const std::vector<double> &query,
                                  std::size_t k,
                                  const SearchParams &params) const
    {
        Clock::time_point start = Clock::now();

        SearchResponse response;
        response.stats.totalClusters = clusters_.size();
        response.stats.totalVectors = vectors_.size();

        if (vectors_.empty() || k == 0)
        {
            // Nothing to search. With k = 0 the empty answer is already exact.
            response.stats.status = vectors_.empty() ? SearchStatus::Exhausted : SearchStatus::Certified;
            response.stats.elapsedMs = millisecondsSince(start);
            return response;
        }

        // 1. Compute every cluster's bound and sort, highest first.
        std::vector<double> q = normalizedQuery(query);
        std::vector<ClusterBound> order = scanOrder(q);
        std::size_t maxClusters = clusterLimit(params, order);

        // 2. Scan clusters in that order, keeping the best k in a heap.
        //    The heap's top is always the worst of the current top k.
        std::priority_queue<Candidate, std::vector<Candidate>, WorstOnTop> topK;
        SearchStatus status = SearchStatus::Exhausted;

        for (std::size_t i = 0; i < order.size(); i++)
        {
            if (i == maxClusters)
            {
                status = SearchStatus::BudgetHit;
                break;
            }

            const Cluster &cluster = clusters_[order[i].cluster];
            bool outOfTime = false;
            std::size_t scannedHere = 0;

            for (std::size_t index : cluster.members)
            {
                double score = dot(q, vectors_[index]);

                if (topK.size() < k)
                {
                    topK.push({score, ids_[index]});
                }
                else
                {
                    // Full: keep the new one only if it beats the current worst.
                    const Candidate &worst = topK.top();
                    if (score > worst.score || (score == worst.score && ids_[index] < worst.id))
                    {
                        topK.pop();
                        topK.push({score, ids_[index]});
                    }
                }
                response.stats.vectorsScanned++;

                // A cluster can be huge, so check the clock inside it too.
                scannedHere++;
                if (scannedHere % kDeadlineCheckStride == 0 && timeIsUp(start, params.deadline))
                {
                    outOfTime = true;
                    break;
                }
            }

            if (outOfTime)
            {
                status = SearchStatus::DeadlineHit; // this cluster is not counted as scanned
                break;
            }

            response.stats.clustersScanned++;

            bool lastCluster = (i + 1 == order.size());
            if (lastCluster)
            {
                break; // scanned everything: status stays Exhausted
            }

            // 3. Can we stop? Every cluster left has a bound no higher than the
            //    next one's. If our k-th best score is above that, nothing left
            //    can get into the top k, so the answer is proven exact.
            double nextBound = order[i + 1].bound;
            if (topK.size() == k && topK.top().score > nextBound + kBoundEpsilon)
            {
                status = SearchStatus::Certified;
                break;
            }

            // Only give up on time once we have something to return.
            if (!topK.empty() && timeIsUp(start, params.deadline))
            {
                status = SearchStatus::DeadlineHit;
                break;
            }
        }

        // 4. Empty the heap. It gives the worst first, so fill from the back.
        response.results.resize(topK.size());
        for (std::size_t i = topK.size(); i > 0; i--)
        {
            response.results[i - 1] = {topK.top().id, topK.top().score};
            topK.pop();
        }

        response.stats.status = status;
        response.stats.elapsedMs = millisecondsSince(start);
        return response;
    }

    // ------------------------------------------------------------------
    // Getters
    // ------------------------------------------------------------------

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
