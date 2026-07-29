#pragma once

#include <vector>
#include <random>
#include <cmath>

namespace vectordb::synthetic
{

    /**
     * Generates random vectors in a unit sphere
     */
    class VectorGenerator
    {
    private:
        std::mt19937 generator_;
        std::uniform_real_distribution<double> distribution_;

    public:
        VectorGenerator(unsigned seed = 42)
            : generator_(seed), distribution_(-1.0, 1.0) {}

        /**
         * Generate a single random vector of given dimension
         */
        std::vector<double> generateRandom(std::size_t dimension)
        {
            std::vector<double> vec(dimension);
            for (std::size_t i = 0; i < dimension; ++i)
            {
                vec[i] = distribution_(generator_);
            }
            return normalize(vec);
        }

        /**
         * Generate multiple random vectors
         */
        std::vector<std::vector<double>> generateRandomVectors(
            std::size_t count, std::size_t dimension)
        {
            std::vector<std::vector<double>> vectors;
            vectors.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
            {
                vectors.push_back(generateRandom(dimension));
            }
            return vectors;
        }

        /**
         * Generate vectors in clusters (for testing nearest neighbor search)
         * Returns: {cluster1_vectors, cluster2_vectors, ...}
         */
        std::vector<std::vector<std::vector<double>>> generateClusters(
            std::size_t num_clusters,
            std::size_t vectors_per_cluster,
            std::size_t dimension,
            double cluster_radius = 0.1)
        {

            std::vector<std::vector<std::vector<double>>> clusters;
            clusters.reserve(num_clusters);

            for (std::size_t c = 0; c < num_clusters; ++c)
            {
                // Generate cluster center
                std::vector<double> center = generateRandom(dimension);

                // Generate vectors around cluster center
                std::vector<std::vector<double>> cluster;
                cluster.reserve(vectors_per_cluster);

                for (std::size_t v = 0; v < vectors_per_cluster; ++v)
                {
                    std::vector<double> noise = generateRandom(dimension);
                    std::vector<double> point(dimension);

                    for (std::size_t d = 0; d < dimension; ++d)
                    {
                        point[d] = center[d] + cluster_radius * noise[d];
                    }
                    cluster.push_back(normalize(point));
                }
                clusters.push_back(cluster);
            }
            return clusters;
        }

    private:
        static std::vector<double> normalize(const std::vector<double> &vec)
        {
            double norm = 0.0;
            for (double val : vec)
            {
                norm += val * val;
            }
            norm = std::sqrt(norm);

            if (norm < 1e-10)
            {
                return vec;
            }

            std::vector<double> normalized(vec.size());
            for (std::size_t i = 0; i < vec.size(); ++i)
            {
                normalized[i] = vec[i] / norm;
            }
            return normalized;
        }
    };

    /**
     * Pre-defined test datasets with known results
     */
    class SyntheticDatasets
    {
    public:
        /**
         * Simple 2D dataset for manual verification
         */
        static std::vector<std::vector<double>> simple2D()
        {
            return {
                {1.0, 0.0},  // vec0
                {0.0, 1.0},  // vec1
                {0.9, 0.1},  // vec2 (close to vec0)
                {0.8, 0.2},  // vec3 (close to vec0)
                {-0.9, 0.1}, // vec4 (opposite to vec0)
                {0.0, -1.0}, // vec5 (opposite to vec1)
            };
        }

        /**
         * 3D dataset with known clusters
         */
        static std::vector<std::vector<double>> simple3D()
        {
            return {
                // Cluster 1: around (1, 0, 0)
                {1.0, 0.0, 0.0},    // vec0
                {0.95, 0.05, 0.05}, // vec1
                {0.9, 0.1, 0.0},    // vec2

                // Cluster 2: around (0, 1, 0)
                {0.0, 1.0, 0.0},    // vec3
                {0.05, 0.95, 0.05}, // vec4
                {0.0, 0.9, 0.1},    // vec5

                // Cluster 3: around (0, 0, 1)
                {0.0, 0.0, 1.0},    // vec6
                {0.05, 0.05, 0.95}, // vec7
                {0.1, 0.0, 0.9},    // vec8
            };
        }

        /**
         * Orthogonal vectors (very easy test case)
         */
        static std::vector<std::vector<double>> orthogonal3D()
        {
            return {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
                {0.0, 0.0, 1.0},
                {-1.0, 0.0, 0.0},
                {0.0, -1.0, 0.0},
                {0.0, 0.0, -1.0},
            };
        }

        /**
         * Identical vectors (edge case)
         */
        static std::vector<std::vector<double>> duplicates()
        {
            return {
                {1.0, 0.0, 0.0},
                {1.0, 0.0, 0.0}, // duplicate
                {0.9, 0.1, 0.0},
                {1.0, 0.0, 0.0}, // duplicate
                {0.8, 0.2, 0.0},
            };
        }
    };

} // namespace vectordb::synthetic
