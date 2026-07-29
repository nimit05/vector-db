#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>

#include "vectordb/collection.hpp"
#include "vectordb/vectorRecord.hpp"
#include "vectordb/synthetic.hpp"

namespace vectordb::benchmark
{

    struct BenchmarkResult
    {
        std::size_t num_vectors;
        std::size_t dimension;
        std::size_t k;
        double insert_time_ms; // Total time to insert all vectors
        double search_time_us; // Time per search operation
        double avg_result_score;

        void print() const
        {
            std::cout << std::setw(12) << num_vectors
                      << std::setw(12) << dimension
                      << std::setw(8) << k
                      << std::setw(15) << std::fixed << std::setprecision(3) << insert_time_ms
                      << std::setw(15) << std::fixed << std::setprecision(3) << search_time_us
                      << std::setw(15) << std::fixed << std::setprecision(6) << avg_result_score
                      << std::endl;
        }

        static void printHeader()
        {
            std::cout << std::setw(12) << "Vectors"
                      << std::setw(12) << "Dimension"
                      << std::setw(8) << "K"
                      << std::setw(15) << "Insert (ms)"
                      << std::setw(15) << "Search (us)"
                      << std::setw(15) << "Avg Score"
                      << std::endl;
            std::cout << std::string(87, '-') << std::endl;
        }
    };

    class BruteForceBenchmark : public ::testing::Test
    {
    protected:
        synthetic::VectorGenerator generator{42};
        std::vector<BenchmarkResult> results;

        void benchmarkInsertAndSearch(std::size_t num_vectors,
                                      std::size_t dimension,
                                      std::size_t k,
                                      int num_searches = 10)
        {
            // Generate vectors
            auto vectors = generator.generateRandomVectors(num_vectors, dimension);

            // Benchmark insertion
            auto col = vectordb::Collection("benchmark");

            auto insert_start = std::chrono::high_resolution_clock::now();
            for (std::size_t i = 0; i < vectors.size(); ++i)
            {
                col.insert(vectordb::VectorRecord("vec" + std::to_string(i), vectors[i]));
            }
            auto insert_end = std::chrono::high_resolution_clock::now();

            double insert_time_ms = std::chrono::duration<double, std::milli>(
                                        insert_end - insert_start)
                                        .count();

            // Benchmark searches
            double total_search_time_us = 0.0;
            double avg_result_score = 0.0;

            for (int s = 0; s < num_searches; ++s)
            {
                auto query = vectors[s % vectors.size()];

                auto search_start = std::chrono::high_resolution_clock::now();
                auto results = col.search(query, k);
                auto search_end = std::chrono::high_resolution_clock::now();

                double search_time_us = std::chrono::duration<double, std::micro>(
                                            search_end - search_start)
                                            .count();
                total_search_time_us += search_time_us;

                // Accumulate average score from first result
                if (!results.empty())
                {
                    avg_result_score += results[0].score;
                }
            }

            double avg_search_time_us = total_search_time_us / num_searches;
            avg_result_score /= num_searches;

            BenchmarkResult result{
                num_vectors, dimension, k,
                insert_time_ms, avg_search_time_us, avg_result_score};
            results.push_back(result);
        }

        void writeResultsToCSV(const std::string &filename)
        {
            std::ofstream csv(filename);
            csv << "num_vectors,dimension,k,insert_time_ms,search_time_us,avg_result_score\n";

            for (const auto &result : results)
            {
                csv << result.num_vectors << ","
                    << result.dimension << ","
                    << result.k << ","
                    << std::fixed << std::setprecision(3) << result.insert_time_ms << ","
                    << std::fixed << std::setprecision(3) << result.search_time_us << ","
                    << std::fixed << std::setprecision(6) << result.avg_result_score << "\n";
            }
            csv.close();
        }
    };

    // ============================================================================
    // Benchmark: Varying Dataset Size (Fixed Dimension and K)
    // ============================================================================

    TEST_F(BruteForceBenchmark, DISABLED_VaryingDatasetSize_128D_K10)
    {
        // Dimension: 128, k: 10
        std::vector<std::size_t> sizes = {100, 500, 1000, 5000, 10000};

        std::cout << "\n=== Brute-Force Benchmark: Varying Dataset Size ===" << std::endl;
        std::cout << "Dimension: 128, K: 10" << std::endl
                  << std::endl;
        BenchmarkResult::printHeader();

        for (std::size_t size : sizes)
        {
            benchmarkInsertAndSearch(size, 128, 10, 10);
            results.back().print();
        }

        std::cout << std::endl;
        writeResultsToCSV("benchmark_varying_size.csv");
    }

    TEST_F(BruteForceBenchmark, DISABLED_VaryingDimension_1000V_K10)
    {
        // Vectors: 1000, k: 10
        std::vector<std::size_t> dimensions = {8, 16, 32, 64, 128, 256, 512};

        std::cout << "\n=== Brute-Force Benchmark: Varying Dimension ===" << std::endl;
        std::cout << "Vectors: 1000, K: 10" << std::endl
                  << std::endl;
        BenchmarkResult::printHeader();

        for (std::size_t dim : dimensions)
        {
            benchmarkInsertAndSearch(1000, dim, 10, 10);
            results.back().print();
        }

        std::cout << std::endl;
        writeResultsToCSV("benchmark_varying_dimension.csv");
    }

    TEST_F(BruteForceBenchmark, DISABLED_VaryingK_1000V_128D)
    {
        // Vectors: 1000, Dimension: 128
        std::vector<std::size_t> ks = {1, 5, 10, 50, 100};

        std::cout << "\n=== Brute-Force Benchmark: Varying K ===" << std::endl;
        std::cout << "Vectors: 1000, Dimension: 128" << std::endl
                  << std::endl;
        BenchmarkResult::printHeader();

        for (std::size_t k : ks)
        {
            benchmarkInsertAndSearch(1000, 128, k, 10);
            results.back().print();
        }

        std::cout << std::endl;
        writeResultsToCSV("benchmark_varying_k.csv");
    }

    TEST_F(BruteForceBenchmark, ComprehensiveBenchmark)
    {
        // Comprehensive benchmark: multiple configs
        std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> configs = {
            // {vectors, dimension, k}
            {100, 64, 10},
            {500, 64, 10},
            {1000, 64, 10},
            {5000, 64, 10},
            {10000, 64, 10},

            {1000, 8, 10},
            {1000, 16, 10},
            {1000, 32, 10},
            {1000, 64, 10},
            {1000, 128, 10},
            {1000, 256, 10},

            {1000, 128, 1},
            {1000, 128, 5},
            {1000, 128, 10},
            {1000, 128, 50},
            {1000, 128, 100},
        };

        std::cout << "\n=== Comprehensive Brute-Force Benchmark ===" << std::endl
                  << std::endl;
        BenchmarkResult::printHeader();

        for (const auto &[n, d, k] : configs)
        {
            benchmarkInsertAndSearch(n, d, k, 10);
            results.back().print();
        }

        std::cout << std::endl;
        writeResultsToCSV("benchmark_comprehensive.csv");

        // Print summary statistics
        std::cout << "\n=== Summary ===" << std::endl;
        double max_search_time = 0.0;
        double min_search_time = 1e9;

        for (const auto &result : results)
        {
            max_search_time = std::max(max_search_time, result.search_time_us);
            min_search_time = std::min(min_search_time, result.search_time_us);
        }

        std::cout << "Min search time: " << min_search_time << " µs" << std::endl;
        std::cout << "Max search time: " << max_search_time << " µs" << std::endl;
        std::cout << "Results saved to benchmark_comprehensive.csv" << std::endl;
    }

    // ============================================================================
    // Quick Baseline Benchmarks (Not Disabled)
    // ============================================================================

    TEST_F(BruteForceBenchmark, QuickBaseline_100Vectors_64D)
    {
        BenchmarkResult::printHeader();
        benchmarkInsertAndSearch(100, 64, 10, 5);
        results.back().print();
    }

    TEST_F(BruteForceBenchmark, QuickBaseline_1000Vectors_64D)
    {
        BenchmarkResult::printHeader();
        benchmarkInsertAndSearch(1000, 64, 10, 5);
        results.back().print();
    }

} // namespace vectordb::benchmark
