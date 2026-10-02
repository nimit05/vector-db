#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "vectordb/collection.hpp"
#include "vectordb/ivf_index.hpp"
#include "vectordb/similarity.hpp"
#include "vectordb/synthetic.hpp"
#include "vectordb/vectorRecord.hpp"

namespace vectordb::test
{

    class IVFIndexTest : public ::testing::Test
    {
    protected:
        std::unordered_map<std::string, VectorRecord> buildRecords(
            const std::vector<std::vector<double>> &vectors)
        {
            std::unordered_map<std::string, VectorRecord> records;
            for (std::size_t i = 0; i < vectors.size(); ++i)
            {
                const std::string id = "vec" + std::to_string(i);
                records.emplace(id, VectorRecord(id, vectors[i]));
            }
            return records;
        }

        Collection buildCollection(const std::string &name,
                                   const std::vector<std::vector<double>> &vectors)
        {
            Collection collection(name);
            for (std::size_t i = 0; i < vectors.size(); ++i)
            {
                collection.insert(VectorRecord("vec" + std::to_string(i), vectors[i]));
            }
            return collection;
        }

        std::vector<std::string> idsOf(const std::vector<SearchResult> &results)
        {
            std::vector<std::string> ids;
            ids.reserve(results.size());
            for (const auto &result : results)
            {
                ids.push_back(result.id);
            }
            return ids;
        }

        std::vector<std::vector<double>> flatten(
            const std::vector<std::vector<std::vector<double>>> &clusters)
        {
            std::vector<std::vector<double>> vectors;
            for (const auto &cluster : clusters)
            {
                vectors.insert(vectors.end(), cluster.begin(), cluster.end());
            }
            return vectors;
        }

        /// Scores from the two paths differ in the last ulp: searchExact divides
        /// raw dot products by the norms, IVF dots pre-normalized copies.
        static constexpr double kScoreTolerance = 1e-12;

        /// The parity invariant: identical ids in identical order under the
        /// (score desc, id asc) total order, scores equal within kScoreTolerance.
        void expectMatchesExact(const std::vector<SearchResult> &ivf,
                                const std::vector<SearchResult> &exact,
                                const std::string &context)
        {
            ASSERT_EQ(ivf.size(), exact.size()) << context;
            EXPECT_EQ(idsOf(ivf), idsOf(exact)) << context;

            for (std::size_t i = 0; i < ivf.size(); ++i)
            {
                EXPECT_NEAR(ivf[i].score, exact[i].score, kScoreTolerance)
                    << context << ", rank " << i;
            }
        }
    };

    // ========================================================================
    // Build: structure
    // ========================================================================

    TEST_F(IVFIndexTest, DefaultNlistIsCeilSqrtN)
    {
        synthetic::VectorGenerator generator(1);
        auto records = buildRecords(generator.generateRandomVectors(100, 16));

        IVFIndex index;
        index.build(records);

        EXPECT_EQ(index.size(), 100);
        EXPECT_EQ(index.clusterCount(), 10); // ceil(sqrt(100))
        EXPECT_EQ(index.dimension(), 16);
    }

    TEST_F(IVFIndexTest, ClustersPartitionEveryVector)
    {
        synthetic::VectorGenerator generator(2);
        auto records = buildRecords(generator.generateRandomVectors(200, 12));

        IVFIndex index;
        index.build(records);

        std::set<std::size_t> seen;
        std::size_t total = 0;

        for (std::size_t c = 0; c < index.clusterCount(); ++c)
        {
            for (std::size_t member : index.clusterMembers(c))
            {
                EXPECT_TRUE(seen.insert(member).second) << "member " << member << " in two clusters";
                ++total;
            }
        }

        EXPECT_EQ(total, index.size());
        EXPECT_EQ(seen.size(), index.size());
    }

    TEST_F(IVFIndexTest, NoClusterIsLeftEmpty)
    {
        synthetic::VectorGenerator generator(3);
        auto records = buildRecords(generator.generateRandomVectors(150, 8));

        IVFIndex index;
        index.build(records);

        for (std::size_t c = 0; c < index.clusterCount(); ++c)
        {
            EXPECT_FALSE(index.clusterMembers(c).empty()) << "cluster " << c << " is empty";
        }
    }

    TEST_F(IVFIndexTest, CentroidsAreUnitNorm)
    {
        synthetic::VectorGenerator generator(4);
        auto records = buildRecords(generator.generateRandomVectors(120, 24));

        IVFIndex index;
        index.build(records);

        for (std::size_t c = 0; c < index.clusterCount(); ++c)
        {
            double norm = 0.0;
            for (double value : index.centroid(c))
            {
                norm += value * value;
            }
            EXPECT_NEAR(std::sqrt(norm), 1.0, 1e-9);
        }
    }

    TEST_F(IVFIndexTest, BuildIsDeterministicForAFixedSeed)
    {
        synthetic::VectorGenerator generator(5);
        auto records = buildRecords(generator.generateRandomVectors(80, 10));

        IVFIndex first;
        IVFIndex second;
        first.build(records, 0, 7);
        second.build(records, 0, 7);

        ASSERT_EQ(first.clusterCount(), second.clusterCount());
        for (std::size_t c = 0; c < first.clusterCount(); ++c)
        {
            EXPECT_EQ(first.clusterMembers(c), second.clusterMembers(c));
        }
    }

    TEST_F(IVFIndexTest, ExplicitNlistIsHonored)
    {
        synthetic::VectorGenerator generator(6);
        auto records = buildRecords(generator.generateRandomVectors(60, 8));

        IVFIndex index;
        index.build(records, 12);

        EXPECT_EQ(index.clusterCount(), 12);
    }

    TEST_F(IVFIndexTest, NlistIsClampedToVectorCount)
    {
        synthetic::VectorGenerator generator(7);
        auto records = buildRecords(generator.generateRandomVectors(5, 4));

        IVFIndex index;
        index.build(records, 50);

        EXPECT_EQ(index.clusterCount(), 5);
        EXPECT_EQ(index.size(), 5);
    }

    TEST_F(IVFIndexTest, ZeroLengthVectorsAreExcluded)
    {
        std::unordered_map<std::string, VectorRecord> records;
        records.emplace("a", VectorRecord("a", {1.0, 0.0, 0.0}));
        records.emplace("b", VectorRecord("b", {0.0, 1.0, 0.0}));
        records.emplace("zero", VectorRecord("zero", {0.0, 0.0, 0.0}));

        IVFIndex index;
        index.build(records);

        EXPECT_EQ(index.size(), 2);

        auto results = index.search({1.0, 0.0, 0.0}, 5, index.clusterCount());
        EXPECT_EQ(results.size(), 2);
        for (const auto &result : results)
        {
            EXPECT_NE(result.id, "zero");
        }
    }

    // ========================================================================
    // Search: correctness against the exact baseline
    // ========================================================================

    TEST_F(IVFIndexTest, FullProbeMatchesExactSearch)
    {
        synthetic::VectorGenerator generator(11);
        auto vectors = flatten(generator.generateClusters(8, 40, 32));
        Collection collection = buildCollection("full_probe", vectors);

        const std::size_t nlist = collection.clusterCount();

        // Queries are both stored vectors and fresh random points, so the test
        // covers queries that sit on a member as well as between clusters.
        auto queries = generator.generateRandomVectors(20, 32);
        for (std::size_t q = 0; q < 20; ++q)
        {
            queries.push_back(vectors[q * 7 % vectors.size()]);
        }

        for (std::size_t q = 0; q < queries.size(); ++q)
        {
            expectMatchesExact(collection.searchIVF(queries[q], 10, nlist),
                               collection.searchExact(queries[q], 10),
                               "query " + std::to_string(q));
        }
    }

    TEST_F(IVFIndexTest, FullProbeMatchesExactSearchOnUnclusteredData)
    {
        synthetic::VectorGenerator generator(11);
        auto vectors = generator.generateRandomVectors(300, 32);
        Collection collection = buildCollection("full_probe_random", vectors);

        const std::size_t nlist = collection.clusterCount();

        for (std::size_t q = 0; q < 20; ++q)
        {
            const auto &query = vectors[q * 7 % vectors.size()];
            expectMatchesExact(collection.searchIVF(query, 10, nlist),
                               collection.searchExact(query, 10),
                               "query " + std::to_string(q));
        }
    }

    TEST_F(IVFIndexTest, ResultsAreSortedDescending)
    {
        synthetic::VectorGenerator generator(12);
        auto vectors = generator.generateRandomVectors(200, 16);
        Collection collection = buildCollection("sorted", vectors);

        auto results = collection.searchIVF(vectors[0], 15, 3);

        ASSERT_FALSE(results.empty());
        for (std::size_t i = 1; i < results.size(); ++i)
        {
            EXPECT_LE(results[i].score, results[i - 1].score);
        }
    }

    TEST_F(IVFIndexTest, SingleProbeFindsTheQueryItself)
    {
        synthetic::VectorGenerator generator(13);
        auto clusters = generator.generateClusters(6, 20, 24, 0.05);

        std::vector<std::vector<double>> vectors;
        for (const auto &cluster : clusters)
        {
            for (const auto &vector : cluster)
            {
                vectors.push_back(vector);
            }
        }

        Collection collection = buildCollection("separated", vectors);

        // On well-separated data a single probe should still surface the exact
        // match, since the query lands squarely inside its own cluster.
        for (std::size_t i = 0; i < vectors.size(); i += 17)
        {
            auto results = collection.searchIVF(vectors[i], 1, 1);
            ASSERT_EQ(results.size(), 1) << "vector " << i;
            EXPECT_GT(results[0].score, 0.99) << "vector " << i;
        }
    }

    TEST_F(IVFIndexTest, MoreProbesNeverLowersTheTopScore)
    {
        synthetic::VectorGenerator generator(14);
        auto vectors = generator.generateRandomVectors(400, 16);
        Collection collection = buildCollection("monotone", vectors);

        const std::size_t nlist = collection.clusterCount();
        auto query = generator.generateRandom(16);

        double previous = -2.0;
        for (std::size_t nprobe = 1; nprobe <= nlist; ++nprobe)
        {
            auto results = collection.searchIVF(query, 5, nprobe);
            ASSERT_FALSE(results.empty());
            EXPECT_GE(results[0].score, previous - 1e-12) << "nprobe " << nprobe;
            previous = results[0].score;
        }
    }

    TEST_F(IVFIndexTest, RespectsK)
    {
        synthetic::VectorGenerator generator(15);
        auto vectors = generator.generateRandomVectors(100, 8);
        Collection collection = buildCollection("k", vectors);

        const std::size_t nlist = collection.clusterCount();

        EXPECT_EQ(collection.searchIVF(vectors[0], 1, nlist).size(), 1);
        EXPECT_EQ(collection.searchIVF(vectors[0], 25, nlist).size(), 25);
        EXPECT_TRUE(collection.searchIVF(vectors[0], 0, nlist).empty());
    }

    TEST_F(IVFIndexTest, KLargerThanCollectionReturnsEverything)
    {
        synthetic::VectorGenerator generator(16);
        auto vectors = generator.generateRandomVectors(20, 8);
        Collection collection = buildCollection("small", vectors);

        auto results = collection.searchIVF(vectors[0], 100, collection.clusterCount());
        EXPECT_EQ(results.size(), 20);
    }

    TEST_F(IVFIndexTest, NprobeIsClampedToClusterCount)
    {
        synthetic::VectorGenerator generator(17);
        auto vectors = generator.generateRandomVectors(150, 12);
        Collection collection = buildCollection("clamp", vectors);

        auto exact = collection.searchExact(vectors[3], 8);
        auto huge = collection.searchIVF(vectors[3], 8, 100000);
        auto zero = collection.searchIVF(vectors[3], 8, 0);

        EXPECT_EQ(idsOf(huge), idsOf(exact));
        EXPECT_EQ(zero.size(), 8); // clamped up to a single probe, not empty
    }

    // ========================================================================
    // Errors and edge cases
    // ========================================================================

    TEST_F(IVFIndexTest, EmptyCollectionReturnsEmpty)
    {
        Collection collection("empty");
        EXPECT_TRUE(collection.searchIVF({1.0, 0.0}, 5, 1).empty());
    }

    TEST_F(IVFIndexTest, DimensionMismatchThrows)
    {
        auto vectors = synthetic::SyntheticDatasets::simple3D();
        Collection collection = buildCollection("dim", vectors);

        EXPECT_THROW(collection.searchIVF({1.0, 0.0}, 3, 1), std::invalid_argument);
    }

    TEST_F(IVFIndexTest, ZeroQueryThrows)
    {
        auto vectors = synthetic::SyntheticDatasets::simple3D();
        Collection collection = buildCollection("zero_query", vectors);

        EXPECT_THROW(collection.searchIVF({0.0, 0.0, 0.0}, 3, 1), std::invalid_argument);
    }

    TEST_F(IVFIndexTest, SingleVectorCollection)
    {
        Collection collection("one");
        collection.insert(VectorRecord("only", {1.0, 2.0, 3.0}));

        auto results = collection.searchIVF({1.0, 2.0, 3.0}, 5, 1);
        ASSERT_EQ(results.size(), 1);
        EXPECT_EQ(results[0].id, "only");
        EXPECT_NEAR(results[0].score, 1.0, 1e-9);
    }

    TEST_F(IVFIndexTest, DuplicateVectorsAllSurface)
    {
        auto vectors = synthetic::SyntheticDatasets::duplicates();
        Collection collection = buildCollection("dupes", vectors);

        auto results = collection.searchIVF({1.0, 0.0, 0.0}, 5, collection.clusterCount());

        int perfectMatches = 0;
        for (const auto &result : results)
        {
            if (std::abs(result.score - 1.0) < 1e-6)
            {
                ++perfectMatches;
            }
        }
        EXPECT_EQ(perfectMatches, 3);
    }

    TEST_F(IVFIndexTest, TiedScoresOrderByIdInBothSearchPaths)
    {
        // vec0, vec1 and vec3 are the same vector, so all three score exactly
        // 1.0 against the query. The id tie-break must order them identically
        // in both paths, or no parity assertion between them is reliable.
        auto vectors = synthetic::SyntheticDatasets::duplicates();
        Collection collection = buildCollection("dupes_order", vectors);

        const std::vector<double> query{1.0, 0.0, 0.0};
        auto exact = collection.searchExact(query, 5);
        auto ivf = collection.searchIVF(query, 5, collection.clusterCount());

        const std::vector<std::string> expected{"vec0", "vec1", "vec3", "vec2", "vec4"};
        EXPECT_EQ(idsOf(exact), expected);
        expectMatchesExact(ivf, exact, "duplicates");
    }

    TEST_F(IVFIndexTest, UnnormalizedRecordsScoreLikeCosine)
    {
        // The index normalizes internally; scores must still match cosine
        // similarity computed on the original, unnormalized vectors.
        Collection collection("scaled");
        collection.insert(VectorRecord("a", {3.0, 0.0, 0.0}));
        collection.insert(VectorRecord("b", {0.0, 7.0, 0.0}));
        collection.insert(VectorRecord("c", {5.0, 5.0, 0.0}));

        auto results = collection.searchIVF({10.0, 0.0, 0.0}, 3, collection.clusterCount());

        ASSERT_EQ(results.size(), 3);
        EXPECT_EQ(results[0].id, "a");
        EXPECT_NEAR(results[0].score, 1.0, 1e-9);
        EXPECT_NEAR(results[1].score, std::sqrt(2.0) / 2.0, 1e-9);
        EXPECT_EQ(results[1].id, "c");
    }

    // ========================================================================
    // Cache invalidation
    // ========================================================================

    TEST_F(IVFIndexTest, IndexRebuildsAfterInsert)
    {
        synthetic::VectorGenerator generator(21);
        auto vectors = generator.generateRandomVectors(50, 8);
        Collection collection = buildCollection("rebuild", vectors);

        auto before = collection.searchIVF(vectors[0], 3, collection.clusterCount());
        ASSERT_FALSE(before.empty());

        collection.insert(VectorRecord("exact_match", vectors[0]));

        auto after = collection.searchIVF(vectors[0], 3, collection.clusterCount());
        ASSERT_FALSE(after.empty());

        auto ids = idsOf(after);
        EXPECT_NE(std::find(ids.begin(), ids.end(), "exact_match"), ids.end())
            << "newly inserted record was not picked up by the index";
    }

    TEST_F(IVFIndexTest, IndexRebuildsAfterRemove)
    {
        synthetic::VectorGenerator generator(22);
        auto vectors = generator.generateRandomVectors(50, 8);
        Collection collection = buildCollection("rebuild_remove", vectors);

        auto before = collection.searchIVF(vectors[0], 1, collection.clusterCount());
        ASSERT_EQ(before.size(), 1);
        const std::string top = before[0].id;

        ASSERT_TRUE(collection.remove(top));

        auto after = collection.searchIVF(vectors[0], 1, collection.clusterCount());
        ASSERT_EQ(after.size(), 1);
        EXPECT_NE(after[0].id, top) << "removed record is still being returned";
    }

    TEST_F(IVFIndexTest, CopiedCollectionRebuildsIndependently)
    {
        synthetic::VectorGenerator generator(23);
        auto vectors = generator.generateRandomVectors(40, 8);
        Collection original = buildCollection("original", vectors);

        // Force the original to build its index, then copy it.
        ASSERT_FALSE(original.searchIVF(vectors[0], 1, 1).empty());

        Collection copy = original;
        copy.insert(VectorRecord("copy_only", vectors[0]));

        auto copyIds = idsOf(copy.searchIVF(vectors[0], 3, copy.clusterCount()));
        auto originalIds = idsOf(original.searchIVF(vectors[0], 3, original.clusterCount()));

        EXPECT_NE(std::find(copyIds.begin(), copyIds.end(), "copy_only"), copyIds.end());
        EXPECT_EQ(std::find(originalIds.begin(), originalIds.end(), "copy_only"), originalIds.end());
    }


    // ========================================================================
    // Provable exactness: the bound and the certificate
    // ========================================================================

    TEST_F(IVFIndexTest, BoundNeverUnderestimatesAMember)
    {
        // The certificate rests on this. If any member scored above its
        // cluster's bound, search could stop before reaching it.
        synthetic::VectorGenerator generator(31);
        const std::vector<std::vector<std::vector<double>>> datasets{
            flatten(generator.generateClusters(6, 30, 16)),
            generator.generateRandomVectors(200, 16),
        };

        for (std::size_t d = 0; d < datasets.size(); ++d)
        {
            const auto &vectors = datasets[d];
            auto records = buildRecords(vectors);

            IVFIndex index;
            index.build(records);

            // Random queries fall between clusters; stored vectors sit inside one.
            auto queries = generator.generateRandomVectors(30, 16);
            for (std::size_t i = 0; i < vectors.size(); i += 11)
            {
                queries.push_back(vectors[i]);
            }

            for (std::size_t q = 0; q < queries.size(); ++q)
            {
                for (std::size_t c = 0; c < index.clusterCount(); ++c)
                {
                    const double bound = index.upperBound(queries[q], c);
                    for (std::size_t member : index.clusterMembers(c))
                    {
                        // The reference score is itself rounded: a query equal
                        // to a member can score an ulp above 1.0 against a bound
                        // of exactly 1.0. Inside search, kBoundEpsilon absorbs that.
                        const auto &values = records.at(index.id(member)).getValues();
                        EXPECT_LE(cosineSimilarity(queries[q], values), bound + kScoreTolerance)
                            << "dataset " << d << ", query " << q << ", cluster " << c;
                    }
                }
            }
        }
    }

    /// k-means seed that recovers the four groups of generateClusters(4, 25, 16)
    /// from VectorGenerator(32), which the well-separated tests below rely on.
    constexpr unsigned kRecoveringSeed = 1;

    TEST_F(IVFIndexTest, WellSeparatedQueryCertifiesAfterOneCluster)
    {
        constexpr std::size_t kGroups = 4;
        constexpr std::size_t kPerGroup = 25;

        synthetic::VectorGenerator generator(32);
        auto vectors = flatten(generator.generateClusters(kGroups, kPerGroup, 16));
        auto records = buildRecords(vectors);
        Collection collection = buildCollection("separated_exact", vectors);

        // Randomly seeded k-means can merge two groups into one cluster; seed 1
        // recovers them. This test is about the bound, not clustering quality.
        IVFIndex index;
        index.build(records, kGroups, kRecoveringSeed);

        // Precondition: k-means recovered the generated groups, one per cluster.
        // "vecN" was generated in group N / kPerGroup.
        for (std::size_t c = 0; c < index.clusterCount(); ++c)
        {
            const auto &members = index.clusterMembers(c);
            ASSERT_EQ(members.size(), kPerGroup) << "cluster " << c;

            const std::size_t group = std::stoul(index.id(members.front()).substr(3)) / kPerGroup;
            for (std::size_t member : members)
            {
                ASSERT_EQ(std::stoul(index.id(member).substr(3)) / kPerGroup, group) << "cluster " << c;
            }
        }

        for (std::size_t i = 0; i < vectors.size(); ++i)
        {
            const SearchResponse response = index.search(vectors[i], 5);
            const std::string context = "query " + std::to_string(i);

            EXPECT_EQ(response.stats.status, SearchStatus::Certified) << context;
            EXPECT_EQ(response.stats.clustersScanned, 1) << context;
            EXPECT_EQ(response.stats.vectorsScanned, kPerGroup) << context;
            expectMatchesExact(response.results, collection.searchExact(vectors[i], 5), context);
        }
    }

    TEST_F(IVFIndexTest, CertificateWaitsForAFullHeap)
    {
        // Same well-separated data, but k exceeds a cluster's size. After the
        // first cluster the heap is not full, there is no k-th score to compare
        // against, and search must keep going however loose the next bound is.
        constexpr std::size_t kGroups = 4;
        constexpr std::size_t kPerGroup = 25;

        synthetic::VectorGenerator generator(32);
        auto vectors = flatten(generator.generateClusters(kGroups, kPerGroup, 16));
        Collection collection = buildCollection("full_heap", vectors);

        IVFIndex index;
        index.build(buildRecords(vectors), kGroups, kRecoveringSeed);
        for (std::size_t c = 0; c < index.clusterCount(); ++c)
        {
            ASSERT_EQ(index.clusterMembers(c).size(), kPerGroup) << "cluster " << c;
        }

        for (std::size_t i = 0; i < vectors.size(); i += 9)
        {
            const SearchResponse response = index.search(vectors[i], kPerGroup + 5);
            const std::string context = "query " + std::to_string(i);

            EXPECT_GE(response.stats.clustersScanned, 2) << context;
            expectMatchesExact(response.results, collection.searchExact(vectors[i], kPerGroup + 5), context);
        }
    }

    TEST_F(IVFIndexTest, ExactSearchMatchesSearchExactOverManyQueries)
    {
        // The load-bearing invariant of the whole index: whenever the status
        // says the answer is exact, it is. A violated bound would not crash or
        // look wrong; it would silently drop a true neighbor. So check it on
        // many queries, several k, and data with ties on the k-th score.
        synthetic::VectorGenerator generator(33);

        // Every vector three times under consecutive ids, so k regularly cuts
        // through a group of exactly tied scores.
        std::vector<std::vector<double>> duplicated;
        for (const auto &vector : flatten(generator.generateClusters(6, 20, 16)))
        {
            duplicated.insert(duplicated.end(), 3, vector);
        }

        struct Dataset
        {
            std::string name;
            std::vector<std::vector<double>> vectors;
            bool expectEarlyStops;
        };

        const std::vector<Dataset> datasets{
            {"clustered", flatten(generator.generateClusters(8, 40, 16)), true},
            {"duplicated", duplicated, true},
            // Unclustered: the bound is loose and may never fire, but whatever
            // the status, exactness must still hold.
            {"random", generator.generateRandomVectors(300, 16), false},
        };

        for (const auto &dataset : datasets)
        {
            Collection collection = buildCollection(dataset.name, dataset.vectors);

            auto queries = generator.generateRandomVectors(60, 16);
            for (std::size_t i = 0; i < dataset.vectors.size(); i += 7)
            {
                queries.push_back(dataset.vectors[i]);
            }

            std::size_t earlyStops = 0;

            for (std::size_t q = 0; q < queries.size(); ++q)
            {
                for (std::size_t k : {1, 5, 20})
                {
                    const SearchResponse response = collection.searchIVF(queries[q], k);
                    const SearchStats &stats = response.stats;
                    const std::string context =
                        dataset.name + ", query " + std::to_string(q) + ", k " + std::to_string(k);

                    ASSERT_TRUE(stats.status == SearchStatus::Certified ||
                                stats.status == SearchStatus::Exhausted)
                        << context;
                    expectMatchesExact(response.results, collection.searchExact(queries[q], k), context);

                    EXPECT_EQ(stats.totalClusters, collection.clusterCount()) << context;
                    EXPECT_EQ(stats.totalVectors, dataset.vectors.size()) << context;
                    EXPECT_LE(stats.vectorsScanned, stats.totalVectors) << context;
                    EXPECT_GE(stats.elapsedMs, 0.0) << context;

                    if (stats.status == SearchStatus::Certified)
                    {
                        EXPECT_LT(stats.clustersScanned, stats.totalClusters) << context;
                        ++earlyStops;
                    }
                    else
                    {
                        EXPECT_EQ(stats.clustersScanned, stats.totalClusters) << context;
                        EXPECT_EQ(stats.vectorsScanned, stats.totalVectors) << context;
                    }
                }
            }

            if (dataset.expectEarlyStops)
            {
                EXPECT_GT(earlyStops, 0) << dataset.name << ": the early-exit path was never exercised";
            }
        }
    }

    TEST_F(IVFIndexTest, ExactSearchEdgeCases)
    {
        Collection empty("empty");
        const SearchResponse nothing = empty.searchIVF({1.0, 0.0}, 3);
        EXPECT_TRUE(nothing.results.empty());
        EXPECT_EQ(nothing.stats.status, SearchStatus::Exhausted);
        EXPECT_EQ(nothing.stats.totalVectors, 0);

        synthetic::VectorGenerator generator(34);
        auto vectors = generator.generateRandomVectors(30, 8);
        Collection collection = buildCollection("edges", vectors);

        // k = 0: the empty answer is exact without scanning anything.
        const SearchResponse none = collection.searchIVF(vectors[0], 0);
        EXPECT_TRUE(none.results.empty());
        EXPECT_EQ(none.stats.status, SearchStatus::Certified);
        EXPECT_EQ(none.stats.clustersScanned, 0);

        // k above the collection size: the heap never fills, so nothing can be
        // certified and every cluster is scanned.
        const SearchResponse all = collection.searchIVF(vectors[0], 1000);
        EXPECT_EQ(all.results.size(), vectors.size());
        EXPECT_EQ(all.stats.status, SearchStatus::Exhausted);

        EXPECT_THROW(collection.searchIVF({1.0, 0.0}, 3), std::invalid_argument);
        EXPECT_THROW(collection.searchIVF(std::vector<double>(8, 0.0), 3), std::invalid_argument);
    }

} // namespace vectordb::test
