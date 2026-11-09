#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "vectordb/collection.hpp"
#include "vectordb/ivf_index.hpp"
#include "vectordb/vectorRecord.hpp"

namespace fs = std::filesystem;

static void printUsage(const std::string &programName)
{
    std::cout << "Usage:\n";
    std::cout << "  " << programName << " insert <collection> <id> <v1> <v2> ...\n";
    std::cout << "  " << programName << " search <collection> <k> <q1> <q2> ... [--mode <mode>]\n";
    std::cout << "  " << programName << " list <collection>\n";
    std::cout << "  " << programName << " delete <collection> <id>\n";
    std::cout << "\n";
    std::cout << "Search modes (IVF index; omit --mode for a brute-force scan):\n";
    std::cout << "  low_latency   1 ms deadline, scans a small share of the clusters\n";
    std::cout << "  balanced      10 ms deadline, scans up to a quarter of the clusters\n";
    std::cout << "  high_recall   100 ms deadline, no cluster cap\n";
    std::cout << "  exact         guaranteed exact; stops early only when the bound proves it\n";
}

static vectordb::SearchMode parseSearchMode(const std::string &name)
{
    if (name == "low_latency")
    {
        return vectordb::SearchMode::LowLatency;
    }
    if (name == "balanced")
    {
        return vectordb::SearchMode::Balanced;
    }
    if (name == "high_recall")
    {
        return vectordb::SearchMode::HighRecall;
    }
    if (name == "exact")
    {
        return vectordb::SearchMode::Exact;
    }
    throw std::invalid_argument("Unknown search mode '" + name +
                                "' (expected low_latency, balanced, high_recall or exact)");
}

static std::string describeStatus(vectordb::SearchStatus status)
{
    switch (status)
    {
    case vectordb::SearchStatus::Certified:
        return "certified";
    case vectordb::SearchStatus::Exhausted:
        return "exhausted";
    case vectordb::SearchStatus::DeadlineHit:
        return "deadline hit";
    case vectordb::SearchStatus::BudgetHit:
        return "budget hit";
    }
    return "unknown";
}

/// e.g. "[certified — scanned 2/32 clusters, 61 vectors, 0.08 ms]"
static std::string formatStats(const vectordb::SearchStats &stats)
{
    std::ostringstream line;
    line << '[' << describeStatus(stats.status)
         << " — scanned " << stats.clustersScanned << '/' << stats.totalClusters << " clusters, "
         << stats.vectorsScanned << " vectors, "
         << std::fixed << std::setprecision(2) << stats.elapsedMs << " ms]";
    return line.str();
}

static std::string getCollectionPath(const std::string &collectionName)
{
    return "data/" + collectionName + ".json";
}

static std::vector<double> parseVectorArgs(int startIndex, int argc, char *argv[])
{
    std::vector<double> values;

    for (int i = startIndex; i < argc; ++i)
    {
        values.push_back(std::stod(argv[i]));
    }

    if (values.empty())
    {
        throw std::invalid_argument("Vector cannot be empty");
    }

    return values;
}

static vectordb::Collection loadOrCreateCollection(const std::string &collectionName)
{
    std::string path = getCollectionPath(collectionName);

    if (fs::exists(path))
    {
        return vectordb::Collection::loadFromFile(path);
    }

    return vectordb::Collection(collectionName);
}

int main(int argc, char *argv[])
{
    try
    {
        if (argc < 3)
        {
            printUsage(argv[0]);
            return 1;
        }

        fs::create_directories("data");

        std::string command = argv[1];
        std::string collectionName = argv[2];
        std::string path = getCollectionPath(collectionName);

        if (command == "insert")
        {
            if (argc < 6)
            {
                std::cerr << "Error: insert requires <collection> <id> <v1> <v2> ...\n";
                printUsage(argv[0]);
                return 1;
            }

            std::string id = argv[3];
            std::vector<double> values = parseVectorArgs(4, argc, argv);

            vectordb::Collection collection = loadOrCreateCollection(collectionName);
            collection.insert(vectordb::VectorRecord(id, values));
            collection.saveToFile(path);

            std::cout << "Inserted record '" << id << "' into collection '" << collectionName << "'.\n";
        }
        else if (command == "search")
        {
            // Pull out "--mode <name>" before anything else: parseVectorArgs reads
            // every remaining argument as a number and would choke on the flag.
            std::optional<vectordb::SearchMode> mode;
            std::vector<char *> args;
            for (int i = 0; i < argc; ++i)
            {
                if (i >= 3 && std::strcmp(argv[i], "--mode") == 0)
                {
                    if (i + 1 >= argc)
                    {
                        throw std::invalid_argument("--mode requires a value");
                    }
                    mode = parseSearchMode(argv[++i]);
                    continue;
                }
                args.push_back(argv[i]);
            }
            argc = static_cast<int>(args.size());
            argv = args.data();

            if (argc < 6)
            {
                std::cerr << "Error: search requires <collection> <k> <q1> <q2> ...\n";
                printUsage(argv[0]);
                return 1;
            }

            if (!fs::exists(path))
            {
                throw std::runtime_error("Collection file does not exist: " + path);
            }

            std::size_t k = static_cast<std::size_t>(std::stoul(argv[3]));
            std::vector<double> query = parseVectorArgs(4, argc, argv);

            vectordb::Collection collection = vectordb::Collection::loadFromFile(path);

            std::vector<vectordb::SearchResult> results;
            std::optional<vectordb::SearchStats> stats;
            if (mode)
            {
                vectordb::SearchResponse response = collection.searchIVF(query, k, *mode);
                results = std::move(response.results);
                stats = response.stats;
            }
            else
            {
                results = collection.searchExact(query, k);
            }

            std::cout << "Top " << k << " results in collection '" << collectionName << "':\n";
            for (const auto &result : results)
            {
                std::cout << "ID: " << result.id
                          << ", Score: " << result.score << '\n';
            }
            if (stats)
            {
                std::cout << formatStats(*stats) << '\n';
            }
        }
        else if (command == "list")
        {
            if (argc != 3)
            {
                std::cerr << "Error: list requires only <collection>\n";
                printUsage(argv[0]);
                return 1;
            }

            if (!fs::exists(path))
            {
                throw std::runtime_error("Collection file does not exist: " + path);
            }

            vectordb::Collection collection = vectordb::Collection::loadFromFile(path);
            auto records = collection.listRecords();

            std::cout << "Records in collection '" << collectionName << "':\n";
            for (const auto &record : records)
            {
                std::cout << record.getId() << '\n';
            }
        }
        else if (command == "delete")
        {
            if (argc != 4)
            {
                std::cerr << "Error: delete requires <collection> <id>\n";
                printUsage(argv[0]);
                return 1;
            }

            if (!fs::exists(path))
            {
                throw std::runtime_error("Collection file does not exist: " + path);
            }

            std::string id = argv[3];

            vectordb::Collection collection = vectordb::Collection::loadFromFile(path);
            bool removed = collection.remove(id);

            if (!removed)
            {
                std::cerr << "Record '" << id << "' not found.\n";
                return 1;
            }

            collection.saveToFile(path);
            std::cout << "Deleted record '" << id << "' from collection '" << collectionName << "'.\n";
        }
        else
        {
            std::cerr << "Error: unknown command '" << command << "'\n";
            printUsage(argv[0]);
            return 1;
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}