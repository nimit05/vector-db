#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
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

        /// True when every cluster holds exactly one generated group, whole.
        /// Assumes ids "vecN" with vector N generated in group N / perGroup.
        bool recoversGroups(const IVFIndex &index, std::size_t groups, std::size_t perGroup)
        {
            if (index.clusterCount() != groups)
            {
                return false;
            }

            for (std::size_t c = 0; c < index.clusterCount(); ++c)
            {
                const auto &members = index.clusterMembers(c);
                if (members.size() != perGroup)
                {
                    return false;
                }

                const std::size_t group = std::stoul(index.id(members.front()).substr(3)) / perGroup;
                for (std::size_t member : members)
                {
                    if (std::stoul(index.id(member).substr(3)) / perGroup != group)
                    {
                        return false;
                    }
                }
            }
            return true;
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
    // k-means++ seeding
    // ========================================================================

    TEST_F(IVFIndexTest, SeedingRecoversSeparatedGroupsAcrossSeeds)
    {
        // Uniform random seeding often put two seeds in one group and none in
        // another, merging two groups into one wide cluster: 28 of these 50
        // seeds recovered the small case and none recovered the larger one.
        // Wide clusters loosen the bound, so this is a search-cost property.
        struct Case
        {
            std::size_t groups;
            std::size_t perGroup;
            std::size_t dimension;
            unsigned dataSeed;
        };

        constexpr unsigned kSeeds = 50;
        constexpr unsigned kRequired = 45;

        for (const Case &test : {Case{4, 25, 16, 32}, Case{16, 50, 32, 7}})
        {
            synthetic::VectorGenerator generator(test.dataSeed);
            auto records = buildRecords(
                flatten(generator.generateClusters(test.groups, test.perGroup, test.dimension)));

            unsigned recovered = 0;
            for (unsigned seed = 0; seed < kSeeds; ++seed)
            {
                IVFIndex index;
                index.build(records, test.groups, seed);
                recovered += recoversGroups(index, test.groups, test.perGroup) ? 1 : 0;
            }

            EXPECT_GE(recovered, kRequired) << test.groups << " groups of " << test.perGroup;
        }
    }

    TEST_F(IVFIndexTest, SeedingHandlesFewerDirectionsThanClusters)
    {
        // Three distinct directions, many copies each, and more clusters than
        // directions. Once every direction has a centroid all distances are
        // zero and there is nothing left to weight a draw by.
        std::vector<std::vector<double>> vectors;
        for (const std::vector<double> &direction :
             {std::vector<double>{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}})
        {
            vectors.insert(vectors.end(), 10, direction);
        }
        Collection collection = buildCollection("few_directions", vectors);

        IVFIndex index;
        index.build(buildRecords(vectors), 6);
        ASSERT_EQ(index.clusterCount(), 6);

        std::size_t total = 0;
        for (std::size_t c = 0; c < index.clusterCount(); ++c)
        {
            total += index.clusterMembers(c).size();
        }
        EXPECT_EQ(total, vectors.size());

        for (std::size_t i = 0; i < vectors.size(); i += 10)
        {
            expectMatchesExact(index.search(vectors[i], 12).results,
                               collection.searchExact(vectors[i], 12),
                               "query " + std::to_string(i));
        }
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

    TEST_F(IVFIndexTest, WellSeparatedQueryCertifiesAfterOneCluster)
    {
        constexpr std::size_t kGroups = 4;
        constexpr std::size_t kPerGroup = 25;

        synthetic::VectorGenerator generator(32);
        auto vectors = flatten(generator.generateClusters(kGroups, kPerGroup, 16));
        auto records = buildRecords(vectors);
        Collection collection = buildCollection("separated_exact", vectors);

        IVFIndex index;
        index.build(records, kGroups);

        // Precondition: k-means recovered the generated groups, one per cluster.
        // This test is about the bound; clustering quality is tested separately.
        ASSERT_TRUE(recoversGroups(index, kGroups, kPerGroup));

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
        index.build(buildRecords(vectors), kGroups);
        ASSERT_TRUE(recoversGroups(index, kGroups, kPerGroup));

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

    // ========================================================================
    // Deadlines, probe budgets and modes
    // ========================================================================

    TEST_F(IVFIndexTest, ModePresetsMatchTheirContracts)
    {
        // Exact is the only mode whose contract rules out a limit being hit, and
        // that is visible in its parameters: nothing to hit.
        const SearchParams exact = paramsForMode(SearchMode::Exact, 10000);
        EXPECT_EQ(exact.deadline.count(), 0);
        EXPECT_EQ(exact.maxProbes, 0);

        // The best-effort modes form a ladder: later modes never scan less or
        // give up sooner than earlier ones.
        const SearchParams low = paramsForMode(SearchMode::LowLatency, 10000);
        const SearchParams balanced = paramsForMode(SearchMode::Balanced, 10000);
        const SearchParams high = paramsForMode(SearchMode::HighRecall, 10000);

        EXPECT_GT(low.deadline.count(), 0);
        EXPECT_LT(low.deadline, balanced.deadline);
        EXPECT_LT(balanced.deadline, high.deadline);

        EXPECT_GE(low.maxProbes, 1);
        EXPECT_LT(low.maxProbes, balanced.maxProbes);
        EXPECT_EQ(high.maxProbes, 0); // uncapped: the clock is its only limit

        // Caps are a share of the cluster count, so they scale with the data,
        // and never fall below a single probe on a tiny collection.
        EXPECT_LT(paramsForMode(SearchMode::Balanced, 100).maxProbes, balanced.maxProbes);
        EXPECT_EQ(paramsForMode(SearchMode::LowLatency, 1).maxProbes, 1);
        EXPECT_EQ(paramsForMode(SearchMode::Balanced, 0).maxProbes, 1);
    }

    TEST_F(IVFIndexTest, ExactModeIsAlwaysExact)
    {
        synthetic::VectorGenerator generator(41);

        for (const auto &vectors : std::vector<std::vector<std::vector<double>>>{
                 flatten(generator.generateClusters(6, 30, 16)),
                 generator.generateRandomVectors(250, 24)})
        {
            Collection collection = buildCollection("exact_mode", vectors);

            auto queries = generator.generateRandomVectors(40, vectors.front().size());
            for (std::size_t i = 0; i < vectors.size(); i += 13)
            {
                queries.push_back(vectors[i]);
            }

            for (std::size_t q = 0; q < queries.size(); ++q)
            {
                const SearchResponse response = collection.searchIVF(queries[q], 10, SearchMode::Exact);
                const std::string context = "query " + std::to_string(q);

                // The contract: Exact never gives up, whatever the clock says.
                ASSERT_TRUE(response.stats.status == SearchStatus::Certified ||
                            response.stats.status == SearchStatus::Exhausted)
                    << context;
                expectMatchesExact(response.results, collection.searchExact(queries[q], 10), context);
            }
        }
    }

    TEST_F(IVFIndexTest, TightDeadlineReturnsBestEffortWithoutThrowing)
    {
        synthetic::VectorGenerator generator(42);
        auto vectors = generator.generateRandomVectors(2000, 32);
        auto records = buildRecords(vectors);

        IVFIndex index;
        index.build(records);

        SearchParams impatient;
        impatient.deadline = std::chrono::microseconds(1);

        SearchParams patient;
        patient.deadline = std::chrono::seconds(10);

        std::size_t deadlineHits = 0;

        for (std::size_t i = 0; i < vectors.size(); i += 211)
        {
            const std::string context = "query " + std::to_string(i);

            const SearchResponse rushed = index.search(vectors[i], 5, impatient);
            // A deadline cuts the scan short; it never returns nothing and never
            // throws, because at least one stride of work always completes.
            ASSERT_FALSE(rushed.results.empty()) << context;
            EXPECT_GT(rushed.stats.vectorsScanned, 0) << context;

            if (rushed.stats.status == SearchStatus::DeadlineHit)
            {
                ++deadlineHits;
                EXPECT_LT(rushed.stats.vectorsScanned, rushed.stats.totalVectors) << context;
            }

            // Given time, the same query on the same index is exact, and the
            // rushed answer's top hit can only be worse than the true one.
            const SearchResponse unhurried = index.search(vectors[i], 5, patient);
            ASSERT_TRUE(unhurried.stats.status == SearchStatus::Certified ||
                        unhurried.stats.status == SearchStatus::Exhausted)
                << context;
            expectMatchesExact(unhurried.results, index.search(vectors[i], 5).results, context);
            EXPECT_LE(rushed.results[0].score, unhurried.results[0].score + kScoreTolerance) << context;

            // The query is itself in the index, so best-bound-first ordering puts
            // it in the very first cluster: the rushed answer still finds it.
            EXPECT_GT(rushed.results[0].score, 0.99) << context;
        }

        EXPECT_GT(deadlineHits, 0) << "a 1us deadline never fired; the clock is not being read";
    }

    TEST_F(IVFIndexTest, DeadlineInterruptsASingleOversizedCluster)
    {
        // With one cluster there is no boundary to check the clock at, so this
        // can only come in under the deadline if the scan reads the clock from
        // inside the inverted list.
        synthetic::VectorGenerator generator(43);
        auto vectors = generator.generateRandomVectors(20000, 64);

        IVFIndex index;
        index.build(buildRecords(vectors), 1);
        ASSERT_EQ(index.clusterCount(), 1);

        SearchParams impatient;
        impatient.deadline = std::chrono::microseconds(1);

        const SearchResponse response = index.search(generator.generateRandom(64), 5, impatient);

        EXPECT_EQ(response.stats.status, SearchStatus::DeadlineHit);
        EXPECT_EQ(response.results.size(), 5);
        EXPECT_LT(response.stats.vectorsScanned, response.stats.totalVectors);
        // The part-scanned cluster is not counted as scanned.
        EXPECT_EQ(response.stats.clustersScanned, 0);
    }

    TEST_F(IVFIndexTest, ProbeBudgetReportsBudgetHit)
    {
        // Unclustered data in a fair number of dimensions: the bound is loose,
        // so the cap is what ends the scan.
        synthetic::VectorGenerator generator(44);
        auto vectors = generator.generateRandomVectors(600, 32);

        IVFIndex index;
        index.build(buildRecords(vectors));
        ASSERT_GT(index.clusterCount(), 2);

        SearchParams capped;
        capped.minProbes = 0; // no floor, so the cap is exactly what is scanned
        capped.maxProbes = 2;

        std::size_t budgetHits = 0;

        for (std::size_t i = 0; i < 25; ++i)
        {
            const SearchResponse response = index.search(generator.generateRandom(32), 5, capped);
            const std::string context = "query " + std::to_string(i);

            EXPECT_LE(response.stats.clustersScanned, capped.maxProbes) << context;
            EXPECT_EQ(response.results.size(), 5) << context;

            if (response.stats.status == SearchStatus::BudgetHit)
            {
                ++budgetHits;
                EXPECT_EQ(response.stats.clustersScanned, capped.maxProbes) << context;
                EXPECT_LT(response.stats.vectorsScanned, response.stats.totalVectors) << context;
            }
            else
            {
                // The only other way out of a capped scan is a proof.
                EXPECT_EQ(response.stats.status, SearchStatus::Certified) << context;
            }
        }

        EXPECT_GT(budgetHits, 0) << "the cap never ended a scan; the budget path is untested";
    }

    TEST_F(IVFIndexTest, ProbeFloorRaisesTheCapButNotTheCertificate)
    {
        synthetic::VectorGenerator generator(45);
        auto vectors = generator.generateRandomVectors(600, 32);

        IVFIndex index;
        index.build(buildRecords(vectors));

        SearchParams floored;
        floored.minProbes = 4;
        floored.maxProbes = 1;

        SearchParams unfloored = floored;
        unfloored.minProbes = 0;

        for (std::size_t i = 0; i < 20; ++i)
        {
            auto query = generator.generateRandom(32);
            const std::string context = "query " + std::to_string(i);

            const SearchResponse wide = index.search(query, 5, floored);
            const SearchResponse narrow = index.search(query, 5, unfloored);

            // The floor lifts the cap it is given, and the extra clusters can
            // only improve the answer. It stays a floor: these bounds are loose
            // enough that the adaptive part raises it further still, and even
            // then the scan stops well short of the whole index.
            EXPECT_EQ(narrow.stats.clustersScanned, 1) << context;
            EXPECT_GE(wide.stats.clustersScanned, floored.minProbes) << context;
            EXPECT_LT(wide.stats.clustersScanned, wide.stats.totalClusters) << context;
            EXPECT_GE(wide.results[0].score, narrow.results[0].score - kScoreTolerance) << context;
        }

        // A floor cannot make an exact search scan past its proof: the whole
        // point is that it acts through the cap only.
        auto vectorsSeparated = flatten(generator.generateClusters(4, 25, 16));
        IVFIndex separated;
        separated.build(buildRecords(vectorsSeparated), 4);
        ASSERT_TRUE(recoversGroups(separated, 4, 25));

        SearchParams highFloor;
        highFloor.minProbes = 4;

        const SearchResponse certified = separated.search(vectorsSeparated[0], 5, highFloor);
        EXPECT_EQ(certified.stats.status, SearchStatus::Certified);
        EXPECT_EQ(certified.stats.clustersScanned, 1);
    }

    TEST_F(IVFIndexTest, AdaptiveFloorProbesMoreForAnAmbiguousQuery)
    {
        // Two tight, well-separated clusters. A query inside one of them has a
        // wide bound gap and needs no second probe; a query halfway between the
        // two is bounded almost equally by both, and the floor should widen the
        // scan even though the cap says one cluster.
        synthetic::VectorGenerator generator(46);
        auto groups = generator.generateClusters(2, 40, 16, 0.02);

        IVFIndex index;
        index.build(buildRecords(flatten(groups)), 2);
        ASSERT_EQ(index.clusterCount(), 2);

        std::vector<double> midpoint(16, 0.0);
        for (std::size_t d = 0; d < midpoint.size(); ++d)
        {
            midpoint[d] = 0.5 * (index.centroid(0)[d] + index.centroid(1)[d]);
        }

        SearchParams onlyOneProbe;
        onlyOneProbe.maxProbes = 1;

        const SearchResponse inside = index.search(index.centroid(0), 5, onlyOneProbe);
        const SearchResponse between = index.search(midpoint, 5, onlyOneProbe);

        EXPECT_EQ(inside.stats.clustersScanned, 1);
        EXPECT_EQ(between.stats.clustersScanned, 2)
            << "the midpoint query should not be resolved on one cluster";
        EXPECT_EQ(between.stats.vectorsScanned, between.stats.totalVectors);
    }

    TEST_F(IVFIndexTest, ModesFormARecallLadder)
    {
        // More permissive modes scan at least as much and score at least as
        // well, because clusters are visited best-bound-first: a longer prefix
        // of the same order can only improve the answer. Timing-dependent
        // outcomes are skipped so this cannot turn flaky on a busy machine.
        synthetic::VectorGenerator generator(47);
        auto vectors = generator.generateRandomVectors(800, 32);
        Collection collection = buildCollection("ladder", vectors);

        const std::vector<SearchMode> ladder{
            SearchMode::LowLatency, SearchMode::Balanced, SearchMode::HighRecall, SearchMode::Exact};

        for (std::size_t i = 0; i < 15; ++i)
        {
            auto query = generator.generateRandom(32);
            const std::string context = "query " + std::to_string(i);

            std::size_t previousClusters = 0;
            double previousScore = -2.0;
            bool comparable = true;

            for (SearchMode mode : ladder)
            {
                const SearchResponse response = collection.searchIVF(query, 5, mode);
                ASSERT_FALSE(response.results.empty()) << context;

                if (comparable)
                {
                    EXPECT_GE(response.stats.clustersScanned, previousClusters) << context;
                    EXPECT_GE(response.results[0].score, previousScore - kScoreTolerance) << context;
                }

                // A deadline makes the next comparison meaningless: the scan was
                // cut by the clock rather than by the mode.
                comparable = response.stats.status != SearchStatus::DeadlineHit;
                previousClusters = response.stats.clustersScanned;
                previousScore = response.results[0].score;
            }
        }
    }

    TEST_F(IVFIndexTest, BestEffortModesStillSurfaceTheObviousAnswer)
    {
        // Every mode, including the most impatient, finds a query that is itself
        // in the index: it sits in the first cluster scanned.
        synthetic::VectorGenerator generator(48);
        auto vectors = flatten(generator.generateClusters(8, 40, 16));
        Collection collection = buildCollection("obvious", vectors);

        for (SearchMode mode : {SearchMode::LowLatency, SearchMode::Balanced,
                                SearchMode::HighRecall, SearchMode::Exact})
        {
            for (std::size_t i = 0; i < vectors.size(); i += 37)
            {
                const SearchResponse response = collection.searchIVF(vectors[i], 1, mode);
                ASSERT_EQ(response.results.size(), 1) << "vector " << i;
                EXPECT_GT(response.results[0].score, 0.99) << "vector " << i;
            }
        }
    }

    TEST_F(IVFIndexTest, DeadlineAndBudgetEdgeCases)
    {
        synthetic::VectorGenerator generator(49);
        auto vectors = generator.generateRandomVectors(200, 8);

        IVFIndex index;
        index.build(buildRecords(vectors));

        SearchParams impatient;
        impatient.deadline = std::chrono::microseconds(1);

        // k = 0 needs no scan at all, so no limit can be hit.
        const SearchResponse none = index.search(vectors[0], 0, impatient);
        EXPECT_TRUE(none.results.empty());
        EXPECT_EQ(none.stats.status, SearchStatus::Certified);

        // An empty index has nothing to scan and nothing to time out of.
        IVFIndex empty;
        const SearchResponse nothing = empty.search({1.0, 0.0}, 5, SearchMode::LowLatency);
        EXPECT_TRUE(nothing.results.empty());
        EXPECT_EQ(nothing.stats.status, SearchStatus::Exhausted);

        // A cap at or above the cluster count is no cap: the scan runs out of
        // clusters before it runs out of budget.
        SearchParams uncapped;
        uncapped.maxProbes = index.clusterCount();
        const SearchResponse full = index.search(vectors[0], 5, uncapped);
        EXPECT_NE(full.stats.status, SearchStatus::BudgetHit);

        // Bad queries are rejected before any clock or budget is consulted.
        EXPECT_THROW(index.search({1.0, 0.0}, 5, impatient), std::invalid_argument);
        EXPECT_THROW(index.search(std::vector<double>(8, 0.0), 5, impatient), std::invalid_argument);
    }

} // namespace vectordb::test
