#ifndef LIBS_LINALG_INCLUDE_LINALG_CSRMATRIX
#define LIBS_LINALG_INCLUDE_LINALG_CSRMATRIX

#include <format>
#include <stdexcept>
#include <vector>

#include <linalg/concepts.h>

namespace linalg
{
    template <typename F>
    struct CsrMatrix
    {
        int rows, cols;

        // Same size (num maybe-nonzero elems)
        std::vector<F> values;
        std::vector<int> colIdx;
        // size = rows + 1, last index is size(coeffs)
        std::vector<int> rowStart;

        void rMult(const F * src, F * dst) const;

        // template <typename A, typename B>
        //     requires VectorLike<A, F> && VectorLike<B, F>
        template <VectorLike<F> A, VectorLike<F> B>
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

        template <VectorLike<F> A, VectorLike<F> B>
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
