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

    /// Adds the record, or replaces the one with the same id. Every vector in a
    /// collection must have the same length; a mismatch throws. The only
    /// exception is replacing the collection's sole record, which may change it.
    void Collection::insert(const VectorRecord &record)
    {
        for (const auto &[id, existing] : records_)
        {
            if (id == record.getId())
            {
                continue; // being replaced, so its length does not count
            }
            if (existing.dimension() != record.dimension())
            {
                throw std::invalid_argument(
                    "Vector length " + std::to_string(record.dimension()) +
                    " does not match collection '" + name_ + "' length " +
                    std::to_string(existing.dimension()));
            }
            break; // all stored vectors share one length, so one check suffices
        }

        records_[record.getId()] = record;

        // add() replaces a vector with the same id, so a replacement is one
        // change, not two. It refuses only what it cannot place (such as the
        // sole record changing dimension); rebuilding handles that.
        if (IVFIndex *index = indexForUpdate())
        {
            if (!index->add(record.getId(), record.getValues()))
            {
                indexDirty_ = true;
            }
        }
    }

    bool Collection::remove(const std::string &id)
    {
        const bool removed = records_.erase(id) > 0;
        if (removed)
        {
            if (IVFIndex *index = indexForUpdate())
            {
                index->remove(id);
            }
        }
        return removed;
    }

    IVFIndex *Collection::indexForUpdate()
    {
        if (indexDirty_ || !index_)
        {
            return nullptr; // the next IVF search rebuilds from the records anyway
        }

        ++changesSinceBuild_;
        if (static_cast<double>(changesSinceBuild_) > kRebuildChurn * static_cast<double>(sizeAtBuild_))
        {
            indexDirty_ = true;
            index_.reset();
            return nullptr;
        }

        // A copy of this collection may share the index; give this one its own
        // before changing it, so the copy keeps searching its own records.
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
        for (const auto &[id, record] : records_)
        {
            records.push_back(record);
        }
        return records;
    }

}