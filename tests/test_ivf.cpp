#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "vectordb/collection.hpp"
#include "vectordb/ivf_index.hpp"
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
        auto vectors = generator.generateRandomVectors(300, 32);
        Collection collection = buildCollection("full_probe", vectors);

        const std::size_t nlist = collection.clusterCount();

        for (std::size_t q = 0; q < 20; ++q)
        {
            const auto &query = vectors[q * 7 % vectors.size()];

            auto exact = collection.searchExact(query, 10);
            auto ivf = collection.searchIVF(query, 10, nlist);

            ASSERT_EQ(ivf.size(), exact.size()) << "query " << q;
            EXPECT_EQ(idsOf(ivf), idsOf(exact)) << "query " << q;

            for (std::size_t i = 0; i < ivf.size(); ++i)
            {
                EXPECT_NEAR(ivf[i].score, exact[i].score, 1e-9);
            }
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

} // namespace vectordb::test
