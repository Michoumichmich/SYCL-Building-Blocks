#pragma once

#include <sycl/sycl.hpp>
#include <type_traits>
#include <array>

namespace sycl::ext {
    namespace runtime_idx_detail {
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
    using can_apply = runtime_idx_detail::can_apply<Z, void, Ts...>;

    template<class T, class Index>
    using subscript_t = decltype(std::declval<T>()[std::declval<Index>()]);

    template<class T, class Index>
    using has_subscript = can_apply<subscript_t, T, Index>;


    namespace runtime_idx_detail {


#define RUNTIME_IDX_STORE_SWITCH_CASE(id, arr, val)\
    case (id):              \
         (arr)[(id)] = val; \
        break;

#define RUNTIME_IDX_STORE_SWITCH_1_CASE(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(0u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_2_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_1_CASE(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(1u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_3_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_2_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(2u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_4_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_3_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(3u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_5_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_4_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(4u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_6_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_5_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(5u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_7_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_6_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(6u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_8_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_7_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(7u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_9_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_8_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(8u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_10_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_9_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(9u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_11_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_10_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(10u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_12_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_11_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(11u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_13_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_12_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(12u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_14_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_13_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(13u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_15_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_14_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(14u, arr, val)
#define RUNTIME_IDX_STORE_SWITCH_16_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_15_CASES(arr, val) RUNTIME_IDX_STORE_SWITCH_CASE(15u, arr, val)

#define GENERATE_IDX_STORE(ID, arr, idx, val)            \
switch(idx){                                            \
    RUNTIME_IDX_STORE_SWITCH_##ID##_CASES(arr, val)     \
      default:                                          \
        (arr)[0] = val;                                 \
}

        template<typename T, typename array_t, int N, int idx_max = N - 1>
        static inline constexpr void runtime_index_wrapper_internal_store(array_t &arr, const int &i, const T val) {
            static_assert(idx_max >= 0 && idx_max < N);
            if constexpr (idx_max == 0 || N == 1) {
                arr[0] = val;
            } else if constexpr (N == 2) {
                GENERATE_IDX_STORE(2, arr, i, val)
            } else if constexpr (N == 3) {
                GENERATE_IDX_STORE(3, arr, i, val)
            } else if constexpr (N == 4) {
                GENERATE_IDX_STORE(4, arr, i, val)
            } else if constexpr (N == 5) {
                GENERATE_IDX_STORE(5, arr, i, val)
            } else if constexpr (N == 6) {
                GENERATE_IDX_STORE(6, arr, i, val)
            } else if constexpr (N == 7) {
                GENERATE_IDX_STORE(7, arr, i, val)
            } else if constexpr (N == 8) {
                GENERATE_IDX_STORE(8, arr, i, val)
            } else if constexpr (N == 9) {
                GENERATE_IDX_STORE(9, arr, i, val)
            } else if constexpr (N == 10) {
                GENERATE_IDX_STORE(10, arr, i, val)
            } else if constexpr (N == 11) {
                GENERATE_IDX_STORE(11, arr, i, val)
            } else if constexpr (N == 12) {
                GENERATE_IDX_STORE(12, arr, i, val)
            } else if constexpr (N == 13) {
                GENERATE_IDX_STORE(13, arr, i, val)
            } else if constexpr (N == 14) {
                GENERATE_IDX_STORE(14, arr, i, val)
            } else if constexpr (N == 15) {
                GENERATE_IDX_STORE(15, arr, i, val)
            } else if constexpr (N == 16) {
                GENERATE_IDX_STORE(16, arr, i, val)
            } else {
                if (i != idx_max) {
                    runtime_index_wrapper_internal_store<T, array_t, N, idx_max - 1>(arr, i, val);
                } else {
                    arr[idx_max] = val;
                }
            }
        }

#undef GENERATE_IDX_STORE

        template<typename T, typename array_t, int N, int idx_max = N - 1>
        static inline constexpr const T &runtime_index_wrapper_internal_read_ref(const array_t &arr, const int &i) {
            static_assert(idx_max >= 0 && idx_max < N);
            if constexpr (idx_max == 0 || N == 1) {
                return arr[0];
            } else {
                if (i == idx_max) {
                    return arr[idx_max];
                } else {
                    return runtime_index_wrapper_internal_read_ref<T, array_t, N, idx_max - 1>(arr, i);
                }
            }
        }

        template<typename T, typename array_t, int N, int idx_max = N - 1>
        static inline constexpr T runtime_index_wrapper_internal_read_copy(const array_t &arr, const int &i) {
            static_assert(idx_max >= 0 && idx_max < N);
            if constexpr (idx_max == 0 || N == 1) {
                return arr[0];
            } else {
                if (i == idx_max) {
                    return arr[idx_max];
                } else {
                    return runtime_index_wrapper_internal_read_copy<T, array_t, N, idx_max - 1>(arr, i);
                }
            }
        }

        template<typename T, typename array_t, size_t N, int end = N - 1, int start = 0>
        static inline constexpr const T &runtime_index_wrapper_log_internal_read_ref(const array_t &array, const int &i) {
            static_assert(start <= end);
            if constexpr (end == start) {
                return array[end];
            } else {
                constexpr int middle = (start + end) / 2;
                static_assert(middle >= 0 && middle < N);
                if (i == middle) {
                    return array[middle];
                } else if (i > middle) {
                    return runtime_index_wrapper_log_internal_read_ref<T, array_t, N, end, middle + 1>(array, i);
                } else {
                    return runtime_index_wrapper_log_internal_read_ref<T, array_t, N, end - 1, start>(array, i);
                }
            }
        }

        template<typename T, typename array_t, size_t N, int end = N - 1, int start = 0>
        static inline constexpr T runtime_index_wrapper_log_internal_read_copy(const array_t &array, const int &i) {
            static_assert(start <= end);
            if constexpr (end == start) {
                return array[end];
            } else {
                constexpr int middle = (start + end) / 2;
                static_assert(middle >= 0 && middle < N);
                if (i == middle) {
                    return array[middle];
                } else if (i > middle) {
                    return runtime_index_wrapper_log_internal_read_copy<T, array_t, N, end, middle + 1>(array, i);
                } else {
                    return runtime_index_wrapper_log_internal_read_copy<T, array_t, N, end - 1, start>(array, i);
                }
            }
        }

    }


/**
 * Subscript operators
 */
    template<int idx_max, typename func, typename T = std::remove_reference_t<subscript_t<func, int>>, typename U>
    static inline constexpr void runtime_index_wrapper(func &f, const int i, const U val) {
        static_assert(has_subscript<func, int>::value, "Must have an int subscript operator");
        static_assert(!std::is_array_v<func>, "Not for arrays");
        runtime_idx_detail::runtime_index_wrapper_internal_store<T, func, idx_max>(f, i, (T) val);
    }

    template<int idx_max, typename func, typename T = std::remove_reference_t<subscript_t<func, int>>>
    static inline constexpr T runtime_index_wrapper(const func &f, const int &i) {
        static_assert(has_subscript<func, int>::value, "Must have an int subscript operator");
        static_assert(!std::is_array_v<func>, "Not for arrays");
        return runtime_idx_detail::runtime_index_wrapper_internal_read_copy<T, func, idx_max>(f, i);
    }


    template<int idx_max, typename func, typename T = std::remove_reference_t<subscript_t<func, int>>>
    static inline constexpr T runtime_index_wrapper_log(const func &f, const int &i) {
        static_assert(has_subscript<func, int>::value, "Must have an int subscript operator");
        static_assert(!std::is_array_v<func>, "Not for arrays");
        return runtime_idx_detail::runtime_index_wrapper_log_internal_read_copy<T, func, idx_max>(f, i);
    }


/**
 * C-Style arrays
 */
    template<typename T, int N, typename U>
    static inline constexpr void runtime_index_wrapper(T (&arr)[N], const int i, const U &val) {
        runtime_idx_detail::runtime_index_wrapper_internal_store<T, T (&)[N], N>(arr, i, (std::remove_reference_t<T>) val);
    }

    template<typename T, int N>
    static inline constexpr const T &runtime_index_wrapper(T const (&arr)[N], const int &i) {
        return runtime_idx_detail::runtime_index_wrapper_internal_read_ref<T, const T (&)[N], N>(arr, i);
    }

    template<typename T, int N>
    static inline constexpr const T &runtime_index_wrapper_log(const T (&arr)[N], const int &i) {
        return runtime_idx_detail::runtime_index_wrapper_log_internal_read_ref<T, const T(&)[N], N>(arr, i);
    }


/**
 * STD::ARRAY
 */
    template<typename T, size_t N, typename U>
    static inline constexpr void runtime_index_wrapper(std::array<T, N> &array, const int i, const U &val) {
        runtime_idx_detail::runtime_index_wrapper_internal_store<T, std::array<T, N>, N>(array, i, (T) val);
    }

    template<typename T, size_t N>
    static inline constexpr const T &runtime_index_wrapper(const std::array<T, N> &array, const int &i) {
        return runtime_idx_detail::runtime_index_wrapper_internal_read_ref<T, std::array<T, N>, N>(array, i);
    }


    template<typename T, size_t N>
    static inline constexpr const T &runtime_index_wrapper_log(const std::array<T, N> &array, const int &i) {
        return runtime_idx_detail::runtime_index_wrapper_log_internal_read_ref<T, std::array<T, N>, N>(array, i);
    }


/**
 * SYCL VEC
 */
    template<template<typename, int> class vec_t, typename T, int N, typename U>
    static inline constexpr void runtime_index_wrapper(vec_t<T, N> &vec, const int i, const U &val) {
        runtime_idx_detail::runtime_index_wrapper_internal_store<T, vec_t<T, N>, N>(vec, i, (T) val);
    }

    template<template<typename, int> class vec_t, typename T, int N>
    static inline constexpr const T &runtime_index_wrapper(const vec_t<T, N> &vec, const int i) {
        return runtime_idx_detail::runtime_index_wrapper_internal_read_ref<T, vec_t<T, N>, N>(vec, i);
    }

    template<template<typename, int> class vec_t, typename T, int N>
    static inline constexpr const T &runtime_index_wrapper_log(const vec_t<T, N> &vec, const int i) {
        return runtime_idx_detail::runtime_index_wrapper_log_internal_read_ref<T, vec_t<T, N>, N>(vec, i);
    }

/**
 * SYCL ID
 */
    template<template<int> class vec_t, int N, typename U>
    static inline constexpr void runtime_index_wrapper(vec_t<N> &vec, const int i, const U &val) {
        runtime_idx_detail::runtime_index_wrapper_internal_store<size_t, vec_t<N>, N>(vec, i, (size_t) val);
    }

    template<template<int> class vec_t, int N>
    static inline constexpr size_t runtime_index_wrapper(const vec_t<N> &vec, const int &i) {
        return runtime_idx_detail::runtime_index_wrapper_internal_read_copy<size_t, vec_t<N>, N>(vec, i);
    }


    template<template<int> class vec_t, int N>
    static inline constexpr size_t runtime_index_wrapper_log(const vec_t<N> &vec, const int &i) {
        return runtime_idx_detail::runtime_index_wrapper_log_internal_read_copy<size_t, vec_t<N>, N>(vec, i);
    }

}