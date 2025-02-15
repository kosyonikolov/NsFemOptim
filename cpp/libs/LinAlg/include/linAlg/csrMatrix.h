#ifndef LIBS_LINALG_INCLUDE_LINALG_CSRMATRIX
#define LIBS_LINALG_INCLUDE_LINALG_CSRMATRIX

#include <concepts>
#include <format>
#include <stdexcept>
#include <vector>

namespace linalg
{
    // clang-format off
    template <typename C, typename T>
    concept VectorLike = requires(C a, const C b) 
    {
        { a.size() } -> std::convertible_to<size_t>;
        { a.data() } -> std::convertible_to<T *>;
        { b.data() } -> std::convertible_to<const T *>;
    };
    // clang-format on

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
    };
} // namespace linalg

#endif /* LIBS_LINALG_INCLUDE_LINALG_CSRMATRIX */
