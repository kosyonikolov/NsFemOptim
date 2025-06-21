#include <cu/cusparse.h>

#include <stdexcept>
#include <format>
#include <iostream>
#include <cassert>

namespace cu
{
    Sparse::Sparse()
    {
        auto rc = cusparseCreate(&theHandle);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to create cusparse: {}", cusparseGetErrorName(rc)));
        }
    }

    Sparse::~Sparse()
    {
        auto rc = cusparseDestroy(theHandle);
        assert(rc == CUSPARSE_STATUS_SUCCESS);
        if (rc != CUSPARSE_STATUS_SUCCESS)
        {
            std::cerr << "Failed to destroy cusparse handle: " << cusparseGetErrorName(rc) << "\n";
        }
    }
}