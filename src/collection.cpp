#include "vectordb/collection.hpp"
#include "vectordb/ivf_index.hpp"
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
        indexDirty_ = true;
    }

    bool Collection::remove(const std::string &id)
    {
        const bool removed = records_.erase(id) > 0;
        if (removed)
        {
            indexDirty_ = true;
        }
        return removed;
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

    std::vector<SearchResult> Collection::searchExact(const std::vector<double> &query, std::size_t k) const
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

        // Score descending, then id ascending. Record iteration order is
        // unspecified, so without the id tie-break tied results would come out
        // in arbitrary order. This is the same total order IVFIndex uses.
        std::sort(results.begin(), results.end(),
                  [](const SearchResult &a, const SearchResult &b)
                  {
                      if (a.score != b.score)
                      {
                          return a.score > b.score;
                      }
                      return a.id < b.id;
                  });

        if (k < results.size())
        {
            results.resize(k);
        }

        return results;
    }

    const IVFIndex &Collection::ivfIndex() const
    {
        if (indexDirty_ || !index_)
        {
            auto index = std::make_shared<IVFIndex>();
            index->build(records_);
            index_ = index;
            indexDirty_ = false;
        }

        return *index_;
    }

    SearchResponse Collection::searchIVF(const std::vector<double> &query, std::size_t k) const
    {
        return ivfIndex().search(query, k);
    }

    std::vector<SearchResult> Collection::searchIVF(const std::vector<double> &query, std::size_t k, std::size_t nprobe) const
    {
        if (records_.empty())
        {
            return {};
        }

        return ivfIndex().search(query, k, nprobe);
    }

    std::size_t Collection::clusterCount() const
    {
        return ivfIndex().clusterCount();
    }

    std::vector<SearchResult> Collection::search(const std::vector<double> &query, std::size_t k) const
    {
        return searchExact(query, k);
    }

    std::vector<VectorRecord> Collection::listRecords() const
    {
        std::vector<VectorRecord> records;
        for (const auto &[id, record] : records_)
        {
            records.push_back(record);
        }
        return records;
    }

}