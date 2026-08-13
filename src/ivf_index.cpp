#include "vectordb/ivf_index.hpp"

#include "vectordb/similarity.hpp"

#include <algorithm>
#include <stdexcept>

namespace vectordb
{

    std::vector<SearchResult> IVFIndex::search(const std::unordered_map<std::string, VectorRecord> &records,
                                               const std::vector<double> &query,
                                               std::size_t k,
                                               std::size_t nprobe) const
    {
        (void)nprobe;

        if (records.empty())
        {
            return {};
        }

        const auto &firstRecord = records.begin()->second;
        if (query.size() != firstRecord.dimension())
        {
            throw std::invalid_argument("Query vector dimension does not match collection vectors");
        }

        std::vector<SearchResult> results;
        results.reserve(records.size());

        for (const auto &[id, record] : records)
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