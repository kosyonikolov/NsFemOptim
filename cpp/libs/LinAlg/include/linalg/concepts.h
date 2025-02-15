#ifndef LIBS_LINALG_INCLUDE_LINALG_CONCEPTS
#define LIBS_LINALG_INCLUDE_LINALG_CONCEPTS

#include <concepts>
#include <cstddef>

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
}

#endif /* LIBS_LINALG_INCLUDE_LINALG_CONCEPTS */
