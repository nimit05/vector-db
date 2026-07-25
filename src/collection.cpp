#include "vectordb/collection.hpp"
#include "vectordb/similarity.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace vectordb
{

    Collection::Collection(const std::string &name) : name_(name)
    {
        if (name_.empty())
        {
            throw std::invalid_argument("Collection name cannot be empty");
        }
    }

    const std::string &Collection::getName() const
    {
        return name_;
    }

    void Collection::insert(const VectorRecord &record)
    {
        records_[record.getId()] = record;
    }

    bool Collection::remove(const std::string &id)
    {
        return records_.erase(id) > 0;
    }

    void Collection::saveToFile(const std::string &filename) const
    {
        json j;
        j["name"] = name_;
        j["records"] = json::array();

        for (const auto &[id, record] : records_)
        {
            j["records"].push_back({{"id", record.getId()},
                                    {"values", record.getValues()}});
        }

        std::ofstream out(filename);
        if (!out.is_open())
        {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        out << j.dump(4);
    }

    Collection Collection::loadFromFile(const std::string &filename)
    {
        std::ifstream in(filename);
        if (!in.is_open())
        {
            throw std::runtime_error("Failed to open file for reading: " + filename);
        }

        json j;
        in >> j;

        Collection collection(j.at("name").get<std::string>());

        for (const auto &item : j.at("records"))
        {
            std::string id = item.at("id").get<std::string>();
            std::vector<double> values = item.at("values").get<std::vector<double>>();
            collection.insert(VectorRecord(id, values));
        }

        return collection;
    }

    std::vector<SearchResult> Collection::search(const std::vector<double> &query, std::size_t k) const
    {
        if (records_.empty())
        {
            return {};
        }

        const auto &firstRecord = records_.begin()->second;
        if (query.size() != firstRecord.dimension())
        {
            throw std::invalid_argument("Query vector dimension does not match collection vectors");
        }

        std::vector<SearchResult> results;
        results.reserve(records_.size());

        for (const auto &[id, record] : records_)
        {
            double score = cosineSimilarity(query, record.getValues());
            results.push_back({id, score});
        }

        std::sort(results.begin(), results.end(),
                  [](const SearchResult &a, const SearchResult &b)
                  {
                      return a.score > b.score;
                  });

        if (k < results.size())
        {
            results.resize(k);
        }

        return results;
    }

}