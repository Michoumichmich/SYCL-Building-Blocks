#pragma once

#include <type_traits>

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

template<int idx_max, typename func, typename T = subscript_t<func, int>>
static inline T &runtime_index_wrapper(func &f, const int i) {
    static_assert(has_subscript<func, int>::value, "Must have an int subscript operator");
    static_assert(!std::is_array_v<func>, "Not for arrays");
    if constexpr (idx_max == 0) {
        return f[idx_max];
    } else {
        if (i == idx_max) {
            return f[idx_max];
        } else {
            return runtime_index_wrapper<idx_max - 1, func, T>(f, i);
        }
    }
}

template<int idx_max, typename func, typename T = subscript_t<func, int>>
static inline T runtime_index_wrapper(const func &f, const int i) {
    static_assert(has_subscript<func, int>::value, "Must have an int subscript operator");
    static_assert(!std::is_array_v<func>, "Not for arrays");
    if constexpr (idx_max == 0) {
        return f[idx_max];
    } else {
        if (i == idx_max) {
            return f[idx_max];
        } else {
            return runtime_index_wrapper<idx_max - 1, func, T>(f, i);
        }
    }
}

template<typename T, int N, int idx_max = N - 1>
static inline T &runtime_index_wrapper(T (&arr)[N], const int i) {
    static_assert(idx_max < N);
    if constexpr (idx_max == 0) {
        return arr[idx_max];
    } else {
        if (i == idx_max) {
            return arr[idx_max];
        } else {
            return runtime_index_wrapper<T, N, idx_max - 1>(arr, i);
        }
    }
}