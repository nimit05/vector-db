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

    namespace
    {
        // Higher score first. Equal scores: smaller id first.
        // IVF search uses the same order, so their results can be compared.
        bool byScoreThenId(const SearchResult &a, const SearchResult &b)
        {
            if (a.score != b.score)
            {
                return a.score > b.score;
            }
            return a.id < b.id;
        }
    }

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

    // Adds a record, or replaces the record with the same id.
    // All vectors must have the same length. The one exception: if the record
    // being replaced is the only one, the new one may have any length.
    void Collection::insert(const VectorRecord &record)
    {
        for (const auto &entry : records_)
        {
            const VectorRecord &existing = entry.second;
            if (existing.getId() == record.getId())
            {
                continue; // this one is being replaced, so skip it
            }
            if (existing.dimension() != record.dimension())
            {
                throw std::invalid_argument(
                    "Vector length " + std::to_string(record.dimension()) +
                    " does not match collection '" + name_ + "' length " +
                    std::to_string(existing.dimension()));
            }
            break; // all vectors have the same length, so checking one is enough
        }

        records_[record.getId()] = record;

        IVFIndex *index = indexForUpdate();
        if (index != nullptr)
        {
            bool added = index->add(record.getId(), record.getValues());
            if (!added)
            {
                indexDirty_ = true; // could not add it, so rebuild instead
            }
        }
    }

    bool Collection::remove(const std::string &id)
    {
        if (records_.erase(id) == 0)
        {
            return false; // no such id
        }

        IVFIndex *index = indexForUpdate();
        if (index != nullptr)
        {
            index->remove(id);
        }
        return true;
    }

    IVFIndex *Collection::indexForUpdate()
    {
        if (index_ == nullptr || indexDirty_)
        {
            return nullptr; // it will be rebuilt from records_ on the next search anyway
        }

        changesSinceBuild_++;
        if (changesSinceBuild_ > kRebuildChurn * sizeAtBuild_)
        {
            // Too many changes: the clusters are getting stale. Rebuild later.
            indexDirty_ = true;
            index_ = nullptr;
            return nullptr;
        }

        // When a Collection is copied, both copies point to the same index.
        // Before changing it, make our own copy so the other one is not affected.
        if (index_.use_count() > 1)
        {
            index_ = std::make_shared<IVFIndex>(*index_);
        }
        return index_.get();
    }

    void Collection::saveToFile(const std::string &filename) const
    {
        json j;
        j["name"] = name_;
        j["records"] = json::array();

        for (const auto &entry : records_)
        {
            const VectorRecord &record = entry.second;
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

        const VectorRecord &first = records_.begin()->second;
        if (query.size() != first.dimension())
        {
            throw std::invalid_argument("Query vector dimension does not match collection vectors");
        }

        // Score every record.
        std::vector<SearchResult> results;
        for (const auto &entry : records_)
        {
            double score = cosineSimilarity(query, entry.second.getValues());
            results.push_back({entry.first, score});
        }

        // Sort best first and keep the top k.
        std::sort(results.begin(), results.end(), byScoreThenId);
        if (k < results.size())
        {
            results.resize(k);
        }
        return results;
    }

    const IVFIndex &Collection::ivfIndex() const
    {
        if (index_ == nullptr || indexDirty_)
        {
            index_ = std::make_shared<IVFIndex>();
            index_->build(records_);
            indexDirty_ = false;
            changesSinceBuild_ = 0;
            sizeAtBuild_ = records_.size();
        }
        return *index_;
    }

    SearchResponse Collection::searchIVF(const std::vector<double> &query, std::size_t k) const
    {
        return ivfIndex().search(query, k);
    }

    SearchResponse Collection::searchIVF(const std::vector<double> &query, std::size_t k, SearchMode mode) const
    {
        return ivfIndex().search(query, k, mode);
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
        for (const auto &entry : records_)
        {
            records.push_back(entry.second);
        }
        return records;
    }

}
