#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <set>

#include "vectordb/collection.hpp"
#include "vectordb/vectorRecord.hpp"
#include "vectordb/synthetic.hpp"

namespace vectordb::test
{

    class SyntheticDataTest : public ::testing::Test
    {
    protected:
        void SetUp() override {}
        void TearDown() override {}

        // Helper to build a collection from vectors
        Collection buildCollection(const std::string &name,
                                   const std::vector<std::vector<double>> &vectors)
        {
            Collection col(name);
            for (std::size_t i = 0; i < vectors.size(); ++i)
            {
                col.insert(VectorRecord("vec" + std::to_string(i), vectors[i]));
            }
            return col;
        }

        // Helper to verify top-k results
        void verifyTopK(const std::vector<SearchResult> &results,
                        const std::set<std::string> &expected_ids,
                        std::size_t k)
        {
            EXPECT_EQ(results.size(), std::min(k, expected_ids.size()));

            std::set<std::string> actual_ids;
            for (const auto &result : results)
            {
                actual_ids.insert(result.id);
            }

            EXPECT_EQ(actual_ids, expected_ids);
        }

        // Helper to verify result is in expected set
        void verifyResultsContain(const std::vector<SearchResult> &results,
                                  const std::set<std::string> &expected_ids)
        {
            std::set<std::string> actual_ids;
            for (const auto &result : results)
            {
                actual_ids.insert(result.id);
            }

            for (const auto &expected : expected_ids)
            {
                EXPECT_NE(actual_ids.find(expected), actual_ids.end())
                    << "Expected to find " << expected << " in results";
            }
        }
    };

    // ============================================================================
    // Tests with Simple 2D Dataset (Manually Verifiable)
    // ============================================================================

    TEST_F(SyntheticDataTest, Simple2DDatasetKnownResults)
    {
        auto vectors = synthetic::SyntheticDatasets::simple2D();
        Collection col = buildCollection("simple2d", vectors);

        // Query: {1, 0} should find {1,0}, {0.9,0.1}, {0.8,0.2}
        std::vector<double> query = {1.0, 0.0};
        auto results = col.search(query, 3);

        EXPECT_EQ(results.size(), 3);
        // Most similar should be the identical vector
        EXPECT_EQ(results[0].id, "vec0");
        EXPECT_NEAR(results[0].score, 1.0, 1e-6);
    }

    TEST_F(SyntheticDataTest, Simple2DLowerScoresAtEnd)
    {
        auto vectors = synthetic::SyntheticDatasets::simple2D();
        Collection col = buildCollection("simple2d", vectors);

        // Query: {1, 0} - results should be sorted by score descending
        std::vector<double> query = {1.0, 0.0};
        auto results = col.search(query, 5);

        // Verify scores are in descending order
        for (std::size_t i = 1; i < results.size(); ++i)
        {
            EXPECT_LE(results[i].score, results[i - 1].score);
        }
    }

    TEST_F(SyntheticDataTest, Simple2DOrthogonalVectors)
    {
        auto vectors = synthetic::SyntheticDatasets::simple2D();
        Collection col = buildCollection("simple2d", vectors);

        // Query: {1, 0} should find {0, 1} with ~0 similarity
        std::vector<double> query = {1.0, 0.0};
        auto results = col.search(query, 5);

        // Find vec1 (0, 1) in results
        auto it = std::find_if(results.begin(), results.end(),
                               [](const SearchResult &r)
                               { return r.id == "vec1"; });
        ASSERT_NE(it, results.end());
        EXPECT_NEAR(it->score, 0.0, 1e-6);
    }

    // ============================================================================
    // Tests with Simple 3D Clustered Dataset
    // ============================================================================

    TEST_F(SyntheticDataTest, Simple3DClustersTopK)
    {
        auto vectors = synthetic::SyntheticDatasets::simple3D();
        Collection col = buildCollection("simple3d", vectors);

        // Query: (1, 0, 0) - should find cluster 1
        std::vector<double> query = {1.0, 0.0, 0.0};
        auto results = col.search(query, 3);

        EXPECT_GE(results.size(), 3);

        // First result should be identical
        EXPECT_EQ(results[0].id, "vec0");
        EXPECT_NEAR(results[0].score, 1.0, 1e-6);

        // Next two should be from same cluster
        std::set<std::string> top3_ids = {results[0].id, results[1].id, results[2].id};
        std::set<std::string> expected = {"vec0", "vec1", "vec2"};
        EXPECT_EQ(top3_ids, expected);
    }

    TEST_F(SyntheticDataTest, Simple3DClusterSeparation)
    {
        auto vectors = synthetic::SyntheticDatasets::simple3D();
        Collection col = buildCollection("simple3d", vectors);

        // Query from cluster 1: (1, 0, 0)
        std::vector<double> query = {1.0, 0.0, 0.0};
        auto results = col.search(query, 9); // Get all

        EXPECT_EQ(results.size(), 9);

        // Verify scores are descending
        for (std::size_t i = 1; i < results.size(); ++i)
        {
            EXPECT_LE(results[i].score, results[i - 1].score);
        }

        // First cluster should dominate top results
        std::set<std::string> top3_ids;
        for (int i = 0; i < 3; ++i)
        {
            top3_ids.insert(results[i].id);
        }

        // All top 3 should be from cluster 1
        for (const auto &id : top3_ids)
        {
            EXPECT_TRUE(id == "vec0" || id == "vec1" || id == "vec2");
        }
    }

    TEST_F(SyntheticDataTest, Simple3DDifferentClusterQueries)
    {
        auto vectors = synthetic::SyntheticDatasets::simple3D();
        Collection col = buildCollection("simple3d", vectors);

        // Query 1: from cluster 1 (1, 0, 0)
        std::vector<double> query1 = {1.0, 0.0, 0.0};
        auto results1 = col.search(query1, 1);
        EXPECT_EQ(results1[0].id, "vec0");

        // Query 2: from cluster 2 (0, 1, 0)
        std::vector<double> query2 = {0.0, 1.0, 0.0};
        auto results2 = col.search(query2, 1);
        EXPECT_EQ(results2[0].id, "vec3");

        // Query 3: from cluster 3 (0, 0, 1)
        std::vector<double> query3 = {0.0, 0.0, 1.0};
        auto results3 = col.search(query3, 1);
        EXPECT_EQ(results3[0].id, "vec6");
    }

    // ============================================================================
    // Tests with Orthogonal Vectors
    // ============================================================================

    TEST_F(SyntheticDataTest, OrthogonalVectorsAllSimilar)
    {
        auto vectors = synthetic::SyntheticDatasets::orthogonal3D();
        Collection col = buildCollection("orthogonal", vectors);

        // Query: (1, 0, 0) - should find itself with score 1.0
        std::vector<double> query = {1.0, 0.0, 0.0};
        auto results = col.search(query, 1);

        EXPECT_EQ(results[0].id, "vec0");
        EXPECT_NEAR(results[0].score, 1.0, 1e-6);
    }

    TEST_F(SyntheticDataTest, OrthogonalVectorsOppositeVector)
    {
        auto vectors = synthetic::SyntheticDatasets::orthogonal3D();
        Collection col = buildCollection("orthogonal", vectors);

        // Query: (1, 0, 0) - (-1, 0, 0) should have score -1.0
        std::vector<double> query = {1.0, 0.0, 0.0};
        auto results = col.search(query, 6); // Get all

        // Find opposite vector
        auto it = std::find_if(results.begin(), results.end(),
                               [](const SearchResult &r)
                               { return r.id == "vec3"; });
        ASSERT_NE(it, results.end());
        EXPECT_NEAR(it->score, -1.0, 1e-6);
        EXPECT_EQ(it, results.end() - 1); // Should be last (lowest score)
    }

    // ============================================================================
    // Tests with Duplicate Vectors (Edge Case)
    // ============================================================================

    TEST_F(SyntheticDataTest, DuplicateVectorsAllMatch)
    {
        auto vectors = synthetic::SyntheticDatasets::duplicates();
        Collection col = buildCollection("duplicates", vectors);

        // Query: (1, 0, 0) - should find all duplicates with score 1.0
        std::vector<double> query = {1.0, 0.0, 0.0};
        auto results = col.search(query, 5);

        // Count how many have score 1.0 (identical)
        int perfect_matches = 0;
        for (const auto &result : results)
        {
            if (std::abs(result.score - 1.0) < 1e-6)
            {
                perfect_matches++;
            }
        }

        // Should have 3 duplicates of (1, 0, 0)
        EXPECT_EQ(perfect_matches, 3);
    }

    // ============================================================================
    // Correctness Tests with Generated Data
    // ============================================================================

    TEST_F(SyntheticDataTest, GeneratedClusters_QueryFindsSameCluster)
    {
        synthetic::VectorGenerator gen(123);

        // Generate 3 clusters of 5 vectors each in 10D space
        auto clusters = gen.generateClusters(3, 5, 10, 0.05);

        // Flatten and build collection
        std::vector<std::vector<double>> all_vectors;
        for (const auto &cluster : clusters)
        {
            for (const auto &vec : cluster)
            {
                all_vectors.push_back(vec);
            }
        }

        Collection col = buildCollection("clusters", all_vectors);

        // Query with cluster 0, vector 0
        auto query = clusters[0][0];
        auto results = col.search(query, 5);

        // Top result should be query itself
        EXPECT_GT(results[0].score, 0.99);
        // Next 4 should mostly be from same cluster
        int same_cluster_count = 0;
        for (int i = 1; i < 5; ++i)
        {
            // Extract vector index from result id
            int result_idx = std::stoi(results[i].id.substr(3));
            // Cluster 0 vectors are at indices 0-4
            if (result_idx < 5)
            {
                same_cluster_count++;
            }
        }

        // At least 3 of top 5 should be from same cluster
        EXPECT_GE(same_cluster_count, 2);
    }

    TEST_F(SyntheticDataTest, RandomVectorsNeverEmpty)
    {
        synthetic::VectorGenerator gen(456);

        for (int dim = 2; dim <= 100; dim *= 2)
        {
            auto vectors = gen.generateRandomVectors(10, dim);

            EXPECT_EQ(vectors.size(), 10);
            for (const auto &vec : vectors)
            {
                EXPECT_EQ(vec.size(), dim);

                // Verify normalized (norm close to 1)
                double norm = 0.0;
                for (double val : vec)
                {
                    norm += val * val;
                }
                norm = std::sqrt(norm);
                EXPECT_NEAR(norm, 1.0, 1e-6);
            }
        }
    }

    TEST_F(SyntheticDataTest, SearchScalabilitySmallData)
    {
        synthetic::VectorGenerator gen(789);

        for (std::size_t n : {10, 50, 100, 500})
        {
            auto vectors = gen.generateRandomVectors(n, 128);
            Collection col = buildCollection("scale_" + std::to_string(n), vectors);

            auto query = vectors[0];
            auto results = col.search(query, 10);

            // Should always find k results (or all if k > n)
            EXPECT_LE(results.size(), std::min(size_t(10), n));
            EXPECT_GE(results.size(), 1);

            // First result should have high score (close to 1.0 due to numerical precision)
            EXPECT_GT(results[0].score, 0.99);
        }
    }

} // namespace vectordb::test
