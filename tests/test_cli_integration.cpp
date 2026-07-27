#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

class CLIIntegrationTest : public ::testing::Test
{
protected:
    std::string cli_path;
    std::string test_data_dir = "cli_test_data";

    void SetUp() override
    {
        // Determine path to CLI executable
        cli_path = "./vectordb";

        // Clean up any previous test data from both directories
        fs::remove_all(test_data_dir);
        fs::remove_all("data");
        fs::create_directories(test_data_dir);
    }

    void TearDown() override
    {
        // Clean up test data from both directories
        fs::remove_all(test_data_dir);
        fs::remove_all("data");
    }

    // Helper to run CLI command and capture output
    std::pair<int, std::string> runCLI(const std::string &args)
    {
        std::string cmd = cli_path + " " + args + " 2>&1";
        FILE *pipe = popen(cmd.c_str(), "r");

        if (!pipe)
        {
            return {-1, "Failed to run command"};
        }

        std::string output;
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
        {
            output += buffer;
        }

        int status = pclose(pipe);
        int exit_code = WEXITSTATUS(status);

        return {exit_code, output};
    }

    // Helper to get collection file path
    std::string getCollectionPath(const std::string &name)
    {
        return "data/" + name + ".json";
    }

    // Helper to check if collection file exists
    bool collectionExists(const std::string &name)
    {
        return fs::exists(getCollectionPath(name));
    }

    // Helper to read collection JSON file
    json readCollectionFile(const std::string &name)
    {
        std::ifstream file(getCollectionPath(name));
        json j;
        file >> j;
        return j;
    }
};

// ============================================================================
// Basic CLI Usage Tests
// ============================================================================

TEST_F(CLIIntegrationTest, CLIWithNoArgumentsShowsUsage)
{
    auto [exit_code, output] = runCLI("");

    EXPECT_NE(exit_code, 0);
    EXPECT_NE(output.find("Usage:"), std::string::npos);
}

TEST_F(CLIIntegrationTest, CLIWithInvalidCommandShowsError)
{
    auto [exit_code, output] = runCLI("invalid_cmd col1");

    EXPECT_NE(exit_code, 0);
    EXPECT_NE(output.find("unknown command"), std::string::npos);
}

// ============================================================================
// Insert Command Tests
// ============================================================================

TEST_F(CLIIntegrationTest, InsertCreatesCollectionFile)
{
    auto [exit_code, output] = runCLI("insert test_col vec1 1.0 0.0 0.0");

    EXPECT_EQ(exit_code, 0);
    EXPECT_NE(output.find("Inserted record"), std::string::npos);
    EXPECT_TRUE(collectionExists("test_col"));
}

TEST_F(CLIIntegrationTest, InsertAddsRecordToCollection)
{
    auto [exit_code, output] = runCLI("insert test_col vec1 1.0 0.0 0.0");
    EXPECT_EQ(exit_code, 0);

    json j = readCollectionFile("test_col");
    EXPECT_EQ(j["name"], "test_col");
    EXPECT_EQ(j["records"].size(), 1);
    EXPECT_EQ(j["records"][0]["id"], "vec1");
    EXPECT_EQ(j["records"][0]["values"][0], 1.0);
}

TEST_F(CLIIntegrationTest, InsertMultipleRecords)
{
    runCLI("insert test_col vec1 1.0 0.0 0.0");
    runCLI("insert test_col vec2 0.0 1.0 0.0");
    runCLI("insert test_col vec3 0.0 0.0 1.0");

    json j = readCollectionFile("test_col");
    EXPECT_EQ(j["records"].size(), 3);
}

TEST_F(CLIIntegrationTest, InsertDuplicateIDReplaces)
{
    runCLI("insert test_col vec1 1.0 0.0 0.0");
    runCLI("insert test_col vec1 2.0 3.0 4.0");

    json j = readCollectionFile("test_col");
    EXPECT_EQ(j["records"].size(), 1);
    EXPECT_EQ(j["records"][0]["values"][0], 2.0);
}

TEST_F(CLIIntegrationTest, InsertWithMissingVectorRejected)
{
    auto [exit_code, output] = runCLI("insert test_col vec1");

    EXPECT_NE(exit_code, 0);
}

TEST_F(CLIIntegrationTest, InsertWithInvalidNumberRejected)
{
    auto [exit_code, output] = runCLI("insert test_col vec1 1.0 invalid 0.0");

    EXPECT_NE(exit_code, 0);
}

// ============================================================================
// Search Command Tests
// ============================================================================

TEST_F(CLIIntegrationTest, SearchFindsNearestNeighbors)
{
    runCLI("insert test_col vec1 1.0 0.0 0.0");
    runCLI("insert test_col vec2 0.8 0.2 0.0");
    runCLI("insert test_col vec3 0.0 1.0 0.0");

    auto [exit_code, output] = runCLI("search test_col 2 1.0 0.0 0.0");

    EXPECT_EQ(exit_code, 0);
    EXPECT_NE(output.find("vec1"), std::string::npos);
    EXPECT_NE(output.find("vec2"), std::string::npos);
}

TEST_F(CLIIntegrationTest, SearchReturnsSortedResults)
{
    runCLI("insert test_col vec1 1.0 0.0 0.0");
    runCLI("insert test_col vec2 0.9 0.1 0.0");
    runCLI("insert test_col vec3 0.0 1.0 0.0");

    auto [exit_code, output] = runCLI("search test_col 3 1.0 0.0 0.0");

    EXPECT_EQ(exit_code, 0);

    // vec1 should appear before vec2, and vec2 before vec3 (by score)
    size_t pos1 = output.find("vec1");
    size_t pos2 = output.find("vec2");
    size_t pos3 = output.find("vec3");

    EXPECT_LT(pos1, pos2);
    EXPECT_LT(pos2, pos3);
}

TEST_F(CLIIntegrationTest, SearchWithKLargerThanCollectionReturnsAll)
{
    runCLI("insert test_col vec1 1.0 0.0 0.0");
    runCLI("insert test_col vec2 0.0 1.0 0.0");

    auto [exit_code, output] = runCLI("search test_col 10 1.0 0.0 0.0");

    EXPECT_EQ(exit_code, 0);
    EXPECT_NE(output.find("Top 10 results"), std::string::npos);
    // Both records should be in output
    EXPECT_NE(output.find("vec1"), std::string::npos);
    EXPECT_NE(output.find("vec2"), std::string::npos);
}

TEST_F(CLIIntegrationTest, SearchNonexistentCollectionFails)
{
    auto [exit_code, output] = runCLI("search nonexistent 1 1.0");

    EXPECT_NE(exit_code, 0);
}

TEST_F(CLIIntegrationTest, SearchWithMismatchedDimensionFails)
{
    runCLI("insert test_col vec1 1.0 0.0 0.0");

    auto [exit_code, output] = runCLI("search test_col 1 1.0 0.0");

    EXPECT_NE(exit_code, 0);
}

TEST_F(CLIIntegrationTest, SearchWithMissingArguments)
{
    auto [exit_code, output] = runCLI("search test_col 1");

    EXPECT_NE(exit_code, 0);
}

// ============================================================================
// List Command Tests
// ============================================================================

TEST_F(CLIIntegrationTest, ListShowsAllRecords)
{
    runCLI("insert test_col vec1 1.0 0.0");
    runCLI("insert test_col vec2 2.0 0.0");
    runCLI("insert test_col vec3 3.0 0.0");

    auto [exit_code, output] = runCLI("list test_col");

    EXPECT_EQ(exit_code, 0);
    EXPECT_NE(output.find("vec1"), std::string::npos);
    EXPECT_NE(output.find("vec2"), std::string::npos);
    EXPECT_NE(output.find("vec3"), std::string::npos);
}

TEST_F(CLIIntegrationTest, ListEmptyCollectionShowsNoRecords)
{
    runCLI("insert test_col vec1 1.0 0.0");
    runCLI("delete test_col vec1");

    auto [exit_code, output] = runCLI("list test_col");

    EXPECT_EQ(exit_code, 0);
    EXPECT_EQ(output.find("vec1"), std::string::npos);
}

TEST_F(CLIIntegrationTest, ListNonexistentCollectionFails)
{
    auto [exit_code, output] = runCLI("list nonexistent");

    EXPECT_NE(exit_code, 0);
    EXPECT_NE(output.find("does not exist"), std::string::npos);
}

TEST_F(CLIIntegrationTest, ListWithExtraArgumentsFails)
{
    runCLI("insert test_col vec1 1.0");

    auto [exit_code, output] = runCLI("list test_col extra");

    EXPECT_NE(exit_code, 0);
}

// ============================================================================
// Delete Command Tests
// ============================================================================

TEST_F(CLIIntegrationTest, DeleteRemovesRecord)
{
    runCLI("insert test_col vec1 1.0 0.0");
    runCLI("insert test_col vec2 0.0 1.0");

    auto [exit_code, output] = runCLI("delete test_col vec1");

    EXPECT_EQ(exit_code, 0);
    EXPECT_NE(output.find("Deleted record"), std::string::npos);

    // Verify deletion
    auto [list_exit, list_output] = runCLI("list test_col");
    EXPECT_EQ(list_output.find("vec1"), std::string::npos);
    EXPECT_NE(list_output.find("vec2"), std::string::npos);
}

TEST_F(CLIIntegrationTest, DeleteNonexistentRecordFails)
{
    runCLI("insert test_col vec1 1.0 0.0");

    auto [exit_code, output] = runCLI("delete test_col nonexistent");

    EXPECT_NE(exit_code, 0);
}

TEST_F(CLIIntegrationTest, DeleteNonexistentCollectionFails)
{
    auto [exit_code, output] = runCLI("delete nonexistent vec1");

    EXPECT_NE(exit_code, 0);
    EXPECT_NE(output.find("does not exist"), std::string::npos);
}

TEST_F(CLIIntegrationTest, DeleteWithMissingIDArgument)
{
    runCLI("insert test_col vec1 1.0");

    auto [exit_code, output] = runCLI("delete test_col");

    EXPECT_NE(exit_code, 0);
}

// ============================================================================
// Integration Tests (Multiple Operations)
// ============================================================================

TEST_F(CLIIntegrationTest, FullWorkflowInsertSearchDelete)
{
    // Insert records
    runCLI("insert vectors vec1 1.0 0.0");
    runCLI("insert vectors vec2 0.9 0.1");
    runCLI("insert vectors vec3 0.0 1.0");

    // Search and verify order
    auto [search_exit, search_output] = runCLI("search vectors 3 1.0 0.0");
    EXPECT_EQ(search_exit, 0);

    // Delete a record
    auto [delete_exit, delete_output] = runCLI("delete vectors vec2");
    EXPECT_EQ(delete_exit, 0);

    // Verify deletion affected search
    auto [search2_exit, search2_output] = runCLI("search vectors 3 1.0 0.0");
    EXPECT_EQ(search2_exit, 0);
    EXPECT_EQ(search2_output.find("vec2"), std::string::npos);
}

TEST_F(CLIIntegrationTest, MultipleCollectionsIndependent)
{
    // Create two separate collections
    runCLI("insert col1 vec1 1.0 0.0");
    runCLI("insert col2 vec1 0.0 1.0");

    // Verify they have different data
    json j1 = readCollectionFile("col1");
    json j2 = readCollectionFile("col2");

    EXPECT_EQ(j1["records"][0]["values"][0], 1.0);
    EXPECT_EQ(j2["records"][0]["values"][0], 0.0);
}

TEST_F(CLIIntegrationTest, PersistenceAcrossInvocations)
{
    // First invocation: insert
    runCLI("insert persist vec1 1.0 0.0 0.0");

    // Second invocation: insert another record
    runCLI("insert persist vec2 0.0 1.0 0.0");

    // Third invocation: verify both exist
    auto [exit, output] = runCLI("list persist");

    EXPECT_NE(output.find("vec1"), std::string::npos);
    EXPECT_NE(output.find("vec2"), std::string::npos);
}

TEST_F(CLIIntegrationTest, LargeVectorHandling)
{
    // Insert a large vector
    std::string large_vec = "insert large_col vec1";
    for (int i = 0; i < 100; ++i)
    {
        large_vec += " " + std::to_string(static_cast<double>(i) / 100.0);
    }

    auto [insert_exit, insert_output] = runCLI(large_vec);
    EXPECT_EQ(insert_exit, 0);

    // Search with matching dimensions
    std::string search_query = "search large_col 1";
    for (int i = 0; i < 100; ++i)
    {
        search_query += " " + std::to_string(static_cast<double>(i) / 100.0);
    }

    auto [search_exit, search_output] = runCLI(search_query);
    EXPECT_EQ(search_exit, 0);
}

TEST_F(CLIIntegrationTest, SpecialCharactersInCollectionName)
{
    auto [exit_code, output] = runCLI("insert testcol_v1 vec1 1.0 0.0");

    EXPECT_EQ(exit_code, 0);
    EXPECT_TRUE(collectionExists("testcol_v1"));
}

// ============================================================================
// Error Handling & Edge Cases
// ============================================================================

TEST_F(CLIIntegrationTest, NegativeVectorValuesAllowed)
{
    auto [exit_code, output] = runCLI("insert test_col vec1 -1.0 -2.0 -3.0");

    EXPECT_EQ(exit_code, 0);
    json j = readCollectionFile("test_col");
    EXPECT_EQ(j["records"][0]["values"][0], -1.0);
}

TEST_F(CLIIntegrationTest, VerySmallVectorValuesAllowed)
{
    auto [exit_code, output] = runCLI("insert test_col vec1 0.0001 0.00001");

    EXPECT_EQ(exit_code, 0);
}

TEST_F(CLIIntegrationTest, ZeroVectorInsertionAllowed)
{
    auto [exit_code, output] = runCLI("insert test_col vec1 0.0 0.0 0.0");

    EXPECT_EQ(exit_code, 0);
}

TEST_F(CLIIntegrationTest, SearchWithZeroVectorFails)
{
    runCLI("insert test_col vec1 1.0 0.0 0.0");

    auto [exit_code, output] = runCLI("search test_col 1 0.0 0.0 0.0");

    EXPECT_NE(exit_code, 0);
    EXPECT_NE(output.find("Zero vector"), std::string::npos);
}

TEST_F(CLIIntegrationTest, SearchResultsIncludeScores)
{
    runCLI("insert test_col vec1 1.0 0.0");
    runCLI("insert test_col vec2 0.0 1.0");

    auto [exit_code, output] = runCLI("search test_col 2 1.0 0.0");

    EXPECT_EQ(exit_code, 0);
    EXPECT_NE(output.find("Score:"), std::string::npos);
}
