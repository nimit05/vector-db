#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace vectordb
{

    class VectorRecord
    {
    private:
        std::string id_;
        std::vector<double> values_;

    public:
        VectorRecord() = default;
        VectorRecord(const std::string &id, const std::vector<double> &values);

        const std::string &getId() const;
        const std::vector<double> &getValues() const;
        std::size_t dimension() const;
    };

}