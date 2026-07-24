#include "vectordb/vectorRecord.hpp"

#include <stdexcept>

namespace vectordb
{

    VectorRecord::VectorRecord(const std::string &id, const std::vector<double> &values)
        : id_(id), values_(values)
    {
        if (id_.empty())
        {
            throw std::invalid_argument("VectorRecord id cannot be empty");
        }

        if (values_.empty())
        {
            throw std::invalid_argument("VectorRecord values cannot be empty");
        }
    }

    const std::string &VectorRecord::getId() const
    {
        return id_;
    }

    const std::vector<double> &VectorRecord::getValues() const
    {
        return values_;
    }

    std::size_t VectorRecord::dimension() const
    {
        return values_.size();
    }

}