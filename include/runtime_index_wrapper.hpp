#pragma once

#include <sycl/sycl.hpp>
#include <type_traits>
#include <array>

namespace sycl::ext {
    namespace details {
        template<class...>
        struct voider {
            using type = void;
        };
        template<class...Ts> using void_t = typename voider<Ts...>::type;

        template<template<class...> class Z, class, class...Ts>
        struct can_apply :
                std::false_type {
        };
        template<template<class...> class Z, class...Ts>
        struct can_apply<Z, void_t<Z<Ts...>>, Ts...> :
                std::true_type {
        };
    }
    template<template<class...> class Z, class...Ts>
    using can_apply = details::can_apply<Z, void, Ts...>;

    template<class T, class Index>
    using subscript_t = decltype(std::declval<T>()[std::declval<Index>()]);

    template<class T, class Index>
    using has_subscript = can_apply<subscript_t, T, Index>;


    namespace detail {
        template<typename T, typename array_t, int N, int idx_max = N - 1, typename idx_t>
        static inline constexpr T &runtime_index_wrapper_internal(array_t &arr, const idx_t &i) {
            static_assert(idx_max < N);
            if constexpr (idx_max == 0) {
                return arr[idx_max];
            } else {
                if (i == idx_max) {
                    return arr[idx_max];
                } else {
                    return runtime_index_wrapper_internal<T, array_t, N, idx_max - 1>(arr, i);
                }
            }
        }


        template<typename T, typename array_t, size_t N, int end = N - 1, int start = 0, typename idx_t>
        static inline constexpr T &runtime_index_wrapper_log_internal(array_t &array, const idx_t &i) {
            static_assert(start <= end);
            if constexpr (end == start) {
                return array[end];
            } else {
                constexpr int mid = (start + end) / 2;
                if (i == mid) {
                    return array[mid];
                } else if (i > mid) {
                    return runtime_index_wrapper_log_internal<T, array_t, N, end, mid + 1>(array, i);
                } else {
                    return runtime_index_wrapper_log_internal<T, array_t, N, end - 1, start>(array, i);
                }
            }
        }
    }


    /**
     * Subscript operators
     */
    template<int idx_max, typename func, typename T = subscript_t<func, int>>
    static inline constexpr T &runtime_index_wrapper(func &f, const int i) {
        static_assert(has_subscript<func, int>::value, "Must have an int subscript operator");
        static_assert(!std::is_array_v<func>, "Not for arrays");
        return detail::runtime_index_wrapper_internal<T, func, idx_max>(f, i);
    }

    template<int idx_max, typename func, typename T = subscript_t<func, int>>
    static inline constexpr T runtime_index_wrapper(const func &f, const int i) {
        static_assert(has_subscript<func, int>::value, "Must have an int subscript operator");
        static_assert(!std::is_array_v<func>, "Not for arrays");
        return detail::runtime_index_wrapper_internal<T, func, idx_max>(f, i);
    }

    template<int idx_max, typename func, typename T = subscript_t<func, int>>
    static inline constexpr T &runtime_index_wrapper_log(func &f, const int i) {
        static_assert(has_subscript<func, int>::value, "Must have an int subscript operator");
        static_assert(!std::is_array_v<func>, "Not for arrays");
        return detail::runtime_index_wrapper_log_internal<T, func, idx_max>(f, i);
    }

    template<int idx_max, typename func, typename T = subscript_t<func, int>>
    static inline constexpr T runtime_index_wrapper_log(const func &f, const int i) {
        static_assert(has_subscript<func, int>::value, "Must have an int subscript operator");
        static_assert(!std::is_array_v<func>, "Not for arrays");
        return detail::runtime_index_wrapper_log_internal<T, func, idx_max>(f, i);
    }


    /**
     * C-Style arrays
     */
    template<typename T, int N>
    static inline constexpr T &runtime_index_wrapper(T (&arr)[N], const int i) {
        return detail::runtime_index_wrapper_internal<T, T (&)[N], N>(arr, i);
    }

    template<typename T, int N>
    static inline constexpr T runtime_index_wrapper(const T (&arr)[N], const int i) {
        return detail::runtime_index_wrapper_internal<T, T (&)[N], N>(arr, i);
    }


    template<typename T, int N>
    static inline constexpr T &runtime_index_wrapper_log(T (&arr)[N], const int i) {
        return detail::runtime_index_wrapper_log_internal<T, T(&)[N], N>(arr, i);
    }

    template<typename T, int N>
    static inline constexpr T runtime_index_wrapper_log(const T (&arr)[N], const int i) {
        return detail::runtime_index_wrapper_log_internal<T, T(&)[N], N>(arr, i);
    }


    /**
     * STD::ARRAY
     */
    template<typename T, size_t N>
    static inline constexpr T &runtime_index_wrapper(std::array<T, N> &array, const size_t i) {
        return detail::runtime_index_wrapper_internal<T, std::array<T, N>, N>(array, i);
    }

    template<typename T, size_t N>
    static inline constexpr T runtime_index_wrapper(const std::array<T, N> &array, const size_t i) {
        return detail::runtime_index_wrapper_internal<T, std::array<T, N>, N>(array, i);
    }

    template<typename T, size_t N>
    static inline constexpr T &runtime_index_wrapper_log(std::array<T, N> &array, const size_t i) {
        return detail::runtime_index_wrapper_log_internal<T, std::array<T, N>, N>(array, i);
    }

    template<typename T, size_t N>
    static inline constexpr T runtime_index_wrapper_log(const std::array<T, N> &array, const size_t i) {
        return detail::runtime_index_wrapper_log_internal<T, std::array<T, N>, N>(array, i);
    }


}