#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <filesystem>

#include "vectordb/collection.hpp"
#include "vectordb/vectorRecord.hpp"
#include "vectordb/similarity.hpp"

namespace fs = std::filesystem;

class VectorRecordTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

class CollectionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        fs::remove_all("test_data");
        fs::create_directories("test_data");
    }

    void TearDown() override
    {
        fs::remove_all("test_data");
    }
};

class SimilarityTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// VectorRecord Tests
// ============================================================================

TEST_F(VectorRecordTest, StoresIDAndVectorCorrectly)
{
    std::string id = "vec1";
    std::vector<double> values = {1.0, 2.0, 3.0};

    vectordb::VectorRecord record(id, values);

    EXPECT_EQ(record.getId(), id);
    EXPECT_EQ(record.getValues(), values);
}

TEST_F(VectorRecordTest, DimensionCalculation)
{
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    vectordb::VectorRecord record("test", values);

    EXPECT_EQ(record.dimension(), 5);
}

TEST_F(VectorRecordTest, DefaultConstructor)
{
    vectordb::VectorRecord record;
    EXPECT_EQ(record.getId(), "");
    EXPECT_EQ(record.getValues().size(), 0);
}

// ============================================================================
// Collection Insert/Delete Tests
// ============================================================================

TEST_F(CollectionTest, InsertAddsRecordSuccessfully)
{
    vectordb::Collection collection("test_col");
    vectordb::VectorRecord record("vec1", {1.0, 0.0, 0.0});

    collection.insert(record);
    auto records = collection.listRecords();

    EXPECT_EQ(records.size(), 1);
    EXPECT_EQ(records[0].getId(), "vec1");
}

TEST_F(CollectionTest, InsertDuplicateIDOverwrites)
{
    vectordb::Collection collection("test_col");
    vectordb::VectorRecord record1("vec1", {1.0, 0.0, 0.0});
    vectordb::VectorRecord record2("vec1", {2.0, 3.0, 4.0});

    collection.insert(record1);
    collection.insert(record2);
    auto records = collection.listRecords();

    EXPECT_EQ(records.size(), 1);
    EXPECT_EQ(records[0].getValues()[0], 2.0);
}

TEST_F(CollectionTest, DeleteRemovesExistingRecord)
{
    vectordb::Collection collection("test_col");
    collection.insert(vectordb::VectorRecord("vec1", {1.0, 0.0, 0.0}));
    collection.insert(vectordb::VectorRecord("vec2", {0.0, 1.0, 0.0}));

    bool removed = collection.remove("vec1");

    EXPECT_TRUE(removed);
    auto records = collection.listRecords();
    EXPECT_EQ(records.size(), 1);
    EXPECT_EQ(records[0].getId(), "vec2");
}

TEST_F(CollectionTest, DeleteMissingIDReturnsFalse)
{
    vectordb::Collection collection("test_col");
    collection.insert(vectordb::VectorRecord("vec1", {1.0, 0.0, 0.0}));

    bool removed = collection.remove("nonexistent");

    EXPECT_FALSE(removed);
    auto records = collection.listRecords();
    EXPECT_EQ(records.size(), 1);
}

TEST_F(CollectionTest, MultipleInsertAndDelete)
{
    vectordb::Collection collection("test_col");

    collection.insert(vectordb::VectorRecord("vec1", {1.0, 0.0, 0.0}));
    collection.insert(vectordb::VectorRecord("vec2", {0.0, 1.0, 0.0}));
    collection.insert(vectordb::VectorRecord("vec3", {0.0, 0.0, 1.0}));

    EXPECT_EQ(collection.listRecords().size(), 3);

    collection.remove("vec2");
    EXPECT_EQ(collection.listRecords().size(), 2);
}

// ============================================================================
// Cosine Similarity Tests
// ============================================================================

TEST_F(SimilarityTest, IdenticalVectorsSimilarityCloseToOne)
{
    std::vector<double> vec = {1.0, 0.0, 0.0};
    double similarity = vectordb::cosineSimilarity(vec, vec);

    EXPECT_NEAR(similarity, 1.0, 1e-9);
}

TEST_F(SimilarityTest, OrthogonalVectorsSimilarityCloseToZero)
{
    std::vector<double> vec1 = {1.0, 0.0, 0.0};
    std::vector<double> vec2 = {0.0, 1.0, 0.0};
    double similarity = vectordb::cosineSimilarity(vec1, vec2);

    EXPECT_NEAR(similarity, 0.0, 1e-9);
}

TEST_F(SimilarityTest, OppositeVectorsSimilarityCloseToNegativeOne)
{
    std::vector<double> vec1 = {1.0, 0.0, 0.0};
    std::vector<double> vec2 = {-1.0, 0.0, 0.0};
    double similarity = vectordb::cosineSimilarity(vec1, vec2);

    EXPECT_NEAR(similarity, -1.0, 1e-9);
}

TEST_F(SimilarityTest, SimilarityIsCommutative)
{
    std::vector<double> vec1 = {1.0, 2.0, 3.0};
    std::vector<double> vec2 = {4.0, 5.0, 6.0};

    double sim1 = vectordb::cosineSimilarity(vec1, vec2);
    double sim2 = vectordb::cosineSimilarity(vec2, vec1);

    EXPECT_NEAR(sim1, sim2, 1e-9);
}

TEST_F(SimilarityTest, EmptyVectorThrows)
{
    std::vector<double> vec1 = {1.0, 0.0, 0.0};
    std::vector<double> empty;

    EXPECT_THROW(vectordb::cosineSimilarity(vec1, empty), std::invalid_argument);
}

TEST_F(SimilarityTest, MismatchedDimensionsThrows)
{
    std::vector<double> vec1 = {1.0, 0.0, 0.0};
    std::vector<double> vec2 = {1.0, 0.0};

    EXPECT_THROW(vectordb::cosineSimilarity(vec1, vec2), std::invalid_argument);
}

TEST_F(SimilarityTest, ZeroVectorThrows)
{
    std::vector<double> vec1 = {1.0, 0.0, 0.0};
    std::vector<double> zero = {0.0, 0.0, 0.0};

    EXPECT_THROW(vectordb::cosineSimilarity(vec1, zero), std::invalid_argument);
}

// ============================================================================
// Search Tests
// ============================================================================

TEST_F(CollectionTest, SearchReturnsTopKInCorrectOrder)
{
    vectordb::Collection collection("test_col");
    collection.insert(vectordb::VectorRecord("vec1", {1.0, 0.0, 0.0}));
    collection.insert(vectordb::VectorRecord("vec2", {0.0, 1.0, 0.0}));
    collection.insert(vectordb::VectorRecord("vec3", {0.8, 0.2, 0.0}));
    collection.insert(vectordb::VectorRecord("vec4", {0.7, 0.3, 0.0}));

    std::vector<double> query = {1.0, 0.0, 0.0};
    auto results = collection.search(query, 2);

    EXPECT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].id, "vec1");
    EXPECT_NEAR(results[0].score, 1.0, 1e-6);
    EXPECT_GT(results[0].score, results[1].score);
}

TEST_F(CollectionTest, SearchWithKLargerThanCollectionReturnsAll)
{
    vectordb::Collection collection("test_col");
    collection.insert(vectordb::VectorRecord("vec1", {1.0, 0.0, 0.0}));
    collection.insert(vectordb::VectorRecord("vec2", {0.0, 1.0, 0.0}));
    collection.insert(vectordb::VectorRecord("vec3", {0.0, 0.0, 1.0}));

    std::vector<double> query = {1.0, 0.0, 0.0};
    auto results = collection.search(query, 10);

    EXPECT_EQ(results.size(), 3);
}

TEST_F(CollectionTest, SearchWithKEqualsCollectionSize)
{
    vectordb::Collection collection("test_col");
    collection.insert(vectordb::VectorRecord("vec1", {1.0, 0.0, 0.0}));
    collection.insert(vectordb::VectorRecord("vec2", {0.0, 1.0, 0.0}));

    std::vector<double> query = {1.0, 0.0, 0.0};
    auto results = collection.search(query, 2);

    EXPECT_EQ(results.size(), 2);
}

TEST_F(CollectionTest, SearchScoresAreDescending)
{
    vectordb::Collection collection("test_col");
    collection.insert(vectordb::VectorRecord("vec1", {1.0, 0.0, 0.0}));
    collection.insert(vectordb::VectorRecord("vec2", {0.9, 0.1, 0.0}));
    collection.insert(vectordb::VectorRecord("vec3", {0.0, 1.0, 0.0}));

    std::vector<double> query = {1.0, 0.0, 0.0};
    auto results = collection.search(query, 3);

    for (size_t i = 1; i < results.size(); ++i)
    {
        EXPECT_GE(results[i - 1].score, results[i].score);
    }
}

TEST_F(CollectionTest, SearchWithMismatchedDimensionThrows)
{
    vectordb::Collection collection("test_col");
    collection.insert(vectordb::VectorRecord("vec1", {1.0, 0.0, 0.0}));

    std::vector<double> query = {1.0, 0.0};

    EXPECT_THROW(collection.search(query, 1), std::invalid_argument);
}

// ============================================================================
// Persistence Tests (Save/Load)
// ============================================================================

TEST_F(CollectionTest, SaveThenLoadPreservesCollectionName)
{
    std::string path = "test_data/collection.json";

    {
        vectordb::Collection collection("my_collection");
        collection.insert(vectordb::VectorRecord("vec1", {1.0, 0.0, 0.0}));
        collection.saveToFile(path);
    }

    vectordb::Collection loaded = vectordb::Collection::loadFromFile(path);
    EXPECT_EQ(loaded.getName(), "my_collection");
}

TEST_F(CollectionTest, SaveThenLoadPreservesRecords)
{
    std::string path = "test_data/collection.json";

    {
        vectordb::Collection collection("test_col");
        collection.insert(vectordb::VectorRecord("vec1", {1.0, 2.0, 3.0}));
        collection.insert(vectordb::VectorRecord("vec2", {4.0, 5.0, 6.0}));
        collection.saveToFile(path);
    }

    vectordb::Collection loaded = vectordb::Collection::loadFromFile(path);
    auto records = loaded.listRecords();

    EXPECT_EQ(records.size(), 2);

    bool found1 = false, found2 = false;
    for (const auto &record : records)
    {
        if (record.getId() == "vec1")
        {
            EXPECT_EQ(record.getValues(), std::vector<double>({1.0, 2.0, 3.0}));
            found1 = true;
        }
        if (record.getId() == "vec2")
        {
            EXPECT_EQ(record.getValues(), std::vector<double>({4.0, 5.0, 6.0}));
            found2 = true;
        }
    }

    EXPECT_TRUE(found1);
    EXPECT_TRUE(found2);
}

TEST_F(CollectionTest, SaveThenLoadPreservesVectorValues)
{
    std::string path = "test_data/collection.json";

    {
        vectordb::Collection collection("test_col");
        collection.insert(vectordb::VectorRecord("vec1", {0.1, 0.2, 0.3, 0.4, 0.5}));
        collection.saveToFile(path);
    }

    vectordb::Collection loaded = vectordb::Collection::loadFromFile(path);
    auto records = loaded.listRecords();

    EXPECT_EQ(records[0].getValues()[0], 0.1);
    EXPECT_EQ(records[0].getValues()[4], 0.5);
}

TEST_F(CollectionTest, SaveThenLoadPreservesSearchBehavior)
{
    std::string path = "test_data/collection.json";

    {
        vectordb::Collection collection("test_col");
        collection.insert(vectordb::VectorRecord("vec1", {1.0, 0.0, 0.0}));
        collection.insert(vectordb::VectorRecord("vec2", {0.8, 0.2, 0.0}));
        collection.insert(vectordb::VectorRecord("vec3", {0.0, 1.0, 0.0}));
        collection.saveToFile(path);
    }

    vectordb::Collection loaded = vectordb::Collection::loadFromFile(path);
    std::vector<double> query = {1.0, 0.0, 0.0};
    auto results = loaded.search(query, 2);

    EXPECT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].id, "vec1");
    EXPECT_NEAR(results[0].score, 1.0, 1e-6);
}

TEST_F(CollectionTest, MultipleRoundTripSaveLoad)
{
    std::string path = "test_data/collection.json";

    // Round 1: Create and save
    {
        vectordb::Collection collection("test_col");
        collection.insert(vectordb::VectorRecord("vec1", {1.0, 0.0}));
        collection.saveToFile(path);
    }

    // Round 2: Load, modify, and save
    {
        vectordb::Collection collection = vectordb::Collection::loadFromFile(path);
        collection.insert(vectordb::VectorRecord("vec2", {0.0, 1.0}));
        collection.saveToFile(path);
    }

    // Round 3: Load and verify
    vectordb::Collection loaded = vectordb::Collection::loadFromFile(path);
    auto records = loaded.listRecords();

    EXPECT_EQ(records.size(), 2);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(CollectionTest, ListRecordsEmptyCollection)
{
    vectordb::Collection collection("test_col");
    auto records = collection.listRecords();

    EXPECT_EQ(records.size(), 0);
}

TEST_F(CollectionTest, CollectionNameCannotBeEmpty)
{
    EXPECT_THROW(vectordb::Collection(""), std::invalid_argument);
}

TEST_F(CollectionTest, SearchEmptyCollection)
{
    vectordb::Collection collection("test_col");
    std::vector<double> query = {1.0, 0.0, 0.0};

    auto results = collection.search(query, 5);
    EXPECT_EQ(results.size(), 0);
}

TEST_F(VectorRecordTest, LargeVectorDimension)
{
    std::vector<double> largeVec(1000);
    for (size_t i = 0; i < 1000; ++i)
    {
        largeVec[i] = static_cast<double>(i) / 1000.0;
    }

    vectordb::VectorRecord record("large", largeVec);
    EXPECT_EQ(record.dimension(), 1000);
    EXPECT_EQ(record.getValues().size(), 1000);
}
