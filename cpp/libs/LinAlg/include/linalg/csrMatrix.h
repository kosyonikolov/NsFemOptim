#ifndef LIBS_LINALG_INCLUDE_LINALG_CSRMATRIX
#define LIBS_LINALG_INCLUDE_LINALG_CSRMATRIX

#include <format>
#include <stdexcept>
#include <vector>
#include <span>

#include <utils/concepts.h>

namespace linalg
{
    template <typename F>
    struct CsrMatrix
    {
        int rows, cols;

        // Same size (num maybe-nonzero elems)
        std::vector<F> values;
        std::vector<int> column;
        // size = rows + 1, last index is size(coeffs)
        std::vector<int> rowStart;

        CsrMatrix<F> slice(std::span<const int> rowIds, std::span<const int> colIds) const;

        template <u::VectorLike<int> A, u::VectorLike<int> B>
        CsrMatrix<F> slice(const A & rowIds, const B & colIds)
        {
            return slice(std::span<const int>(rowIds.data(), rowIds.size()), std::span<const int>(colIds.data(), colIds.size()));
        }

        // Find indices in values for the given row/col pairs
        // Non-found values are set to -1
        // Column ids MUST be sorted
        void findOffsets(const int row, std::span<const int> columnIds, std::span<int> dstOffsets) const;

        template <u::VectorLike<int> A, u::VectorLike<int> B>
        void findOffsets(const int row, const A & columnIds, B & dstOffsets) const
        {
            return findOffsets(row, std::span<const int>(columnIds.data(), columnIds.size()), std::span<int>(dstOffsets.data(), dstOffsets.size()));
        }

        void findOffsetsUnsorted(const int row, std::span<const int> columnIds, std::span<int> dstOffsets) const;

        template <u::VectorLike<int> A, u::VectorLike<int> B>
        void findOffsetsUnsorted(const int row, const A & columnIds, B & dstOffsets) const
        {
            return findOffsetsUnsorted(row, std::span<const int>(columnIds.data(), columnIds.size()), std::span<int>(dstOffsets.data(), dstOffsets.size()));
        }

        bool compareLayout(const CsrMatrix<F> & other) const;

        bool compareValues(const CsrMatrix<F> & other, const F epsilon) const;

        bool operator==(const CsrMatrix<F> & other) const;

        bool operator!=(const CsrMatrix<F> & other) const;

        void rMult(const F * src, F * dst) const;

        // template <typename A, typename B>
        //     requires VectorLike<A, F> && VectorLike<B, F>
        template <u::VectorLike<F> A, u::VectorLike<F> B>
        void rMult(const A & src, B & dst) const
        {
            if (src.size() != cols)
            {
                throw std::invalid_argument(std::format("{}: Bad size of src vector [{}] - expected {}",
                                                        __FUNCTION__, src.size(), cols));
            }
            if (dst.size() != rows)
            {
                throw std::invalid_argument(std::format("{}: Bad size of dst vector [{}] - expected {}",
                                                        __FUNCTION__, dst.size(), rows));
            }
            rMult(src.data(), dst.data());
        }

        // sqrt((Mx - b).^2 / rows)
        double mse(const F * x, const F * b) const;

        template <u::VectorLike<F> A, u::VectorLike<F> B>
        double mse(const A & x, const B & b) const
        {
            if (x.size() != cols)
            {
                throw std::invalid_argument(std::format("{}: Bad size of x vector [{}] - expected {}",
                                                        __FUNCTION__, x.size(), cols));
            }
            if (b.size() != rows)
            {
                throw std::invalid_argument(std::format("{}: Bad size of dst vector [{}] - expected {}",
                                                        __FUNCTION__, b.size(), rows));
            }
            return mse(x.data(), b.data());
        }
    };
} // namespace linalg

#endif /* LIBS_LINALG_INCLUDE_LINALG_CSRMATRIX */
