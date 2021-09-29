/**
    Copyright 2021 Codeplay Software Ltd.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use these files except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    For your convenience, a copy of the License has been included in this
    repository.

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    @author Michel Migdal.
 */


#pragma once

#include <sycl/sycl.hpp>
#include <type_traits>
#include <array>

namespace sycl::ext {
    namespace registerizer_internal {
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
    using can_apply = registerizer_internal::can_apply<Z, void, Ts...>;

    template<class T, class Index>
    using subscript_t = decltype(std::declval<T>()[std::declval<Index>()]);

    template<class T, class Index>
    using has_subscript = can_apply<subscript_t, T, Index>;


    namespace registerizer_internal {


#ifdef RUNTIME_IDX_STORE_USE_SWITCH
#define RUNTIME_IDX_STORE_SWITCH_CASE(id, arr, val)\
        case (id): (arr)[(id)] = val; break;
#else
#define RUNTIME_IDX_STORE_SWITCH_CASE(id, arr, val)\
    (arr)[(id)] = ((id)==(i)) ? (val) : (arr)[(id)] ;

#endif

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

#ifdef RUNTIME_IDX_STORE_USE_SWITCH
#define GENERATE_IDX_STORE(ID, arr, idx, val)            \
switch(idx){                                            \
    RUNTIME_IDX_STORE_SWITCH_##ID##_CASES(arr, val)     \
      default:                                          \
        (arr)[0] = val;                                 \
}
#else
#define GENERATE_IDX_STORE(ID, arr, idx, val)            \
{                                                       \
    RUNTIME_IDX_STORE_SWITCH_##ID##_CASES(arr, val)     \
}
#endif


        template<typename T, typename array_t, int N, int idx_max = N - 1>
        static inline constexpr void registerized_store(array_t &arr, const uint &i, const T &val) noexcept {
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
            } else if constexpr (N == 16 || idx_max == 15) { // End of recursion if we started with N>16
                GENERATE_IDX_STORE(16, arr, i, val)
            } else {
                if (i != idx_max) {
                    registerized_store<T, array_t, N, idx_max - 1>(arr, i, val);
                } else {
                    arr[idx_max] = val;
                }
            }
        }

#undef GENERATE_IDX_STORE

        template<typename func, typename T, typename array_t, int N, int idx_max = N - 1>
        static inline constexpr void registerized_const_forall(const array_t &arr, const func &&f) noexcept {
            static_assert(idx_max >= 0 && idx_max < N);
#pragma unroll
            for (uint i = 0; i < N; ++i) {
                f(i, arr[i]);
            }
        }

        template<typename func, typename T, typename array_t, int N, int idx_max = N - 1>
        static inline constexpr void registerize_transform_ith(array_t &arr, const func &&f, const uint &idx) noexcept {
            static_assert(idx_max >= 0 && idx_max < N);
#pragma unroll
            for (uint i = 0; i < N; ++i) {
                arr[i] = (idx == i) ? f(arr[i]) : arr[i];
            }
        }


        template<typename T, typename array_t, int N, int idx_max = N - 1>
        [[nodiscard]] static inline constexpr T registerized_read(const array_t &arr, const uint &idx) noexcept {
            static_assert(idx_max >= 0 && idx_max < N);
            if constexpr (idx_max == 0 || N == 1) {
                return arr[0];
            } else {
                if (idx == idx_max) {
                    return arr[idx_max];
                } else {
                    return registerized_read<T, array_t, N, idx_max - 1>(arr, idx);
                }
            }
        }

        template<typename T, typename array_t, size_t N, int end = N - 1, int start = 0>
        [[nodiscard]] static inline constexpr T registerized_dicochotomic_read(const array_t &array, const uint &idx) noexcept {
            static_assert(start <= end);
            if constexpr (end == start) {
                return array[end];
            } else {
                constexpr int middle = (start + end) / 2;
                static_assert(middle >= 0 && middle < N);
                if (idx == middle) {
                    return array[middle];
                } else if (idx > middle) {
                    return registerized_dicochotomic_read<T, array_t, N, end, middle + 1>(array, idx);
                } else {
                    return registerized_dicochotomic_read<T, array_t, N, end - 1, start>(array, idx);
                }
            }
        }

    }


/**
 * Subscript operators
 */
    template<int idx_max, typename func, typename T = std::remove_reference_t<subscript_t<func, int>>, typename U>
    static inline constexpr U runtime_index_wrapper(func &f, const uint &i, const U &val) {
        static_assert(has_subscript<func, int>::value, "Must have an int subscript operator");
        static_assert(!std::is_array_v<func>, "Not for arrays");
        registerizer_internal::registerized_store<T, func, idx_max>(f, i, (T) val);
        return val;
    }

    template<int idx_max, typename func, typename T = std::remove_reference_t<subscript_t<func, int>>>
    [[nodiscard]] static inline constexpr T runtime_index_wrapper(const func &f, const uint &i) {
        static_assert(has_subscript<func, int>::value, "Must have an int subscript operator");
        static_assert(!std::is_array_v<func>, "Not for arrays");
        return registerizer_internal::registerized_read<T, func, idx_max>(f, i);
    }


    template<int idx_max, typename func, typename T = std::remove_reference_t<subscript_t<func, int>>>
    [[nodiscard]] static inline constexpr T runtime_index_wrapper_log(const func &f, const uint &i) {
        static_assert(has_subscript<func, int>::value, "Must have an int subscript operator");
        static_assert(!std::is_array_v<func>, "Not for arrays");
        return registerizer_internal::registerized_dicochotomic_read<T, func, idx_max>(f, i);
    }


/**
 * C-Style arrays
 */
    template<typename T, int N, typename U>
    static inline constexpr U runtime_index_wrapper(T (&arr)[N], const uint i, const U &val) {
        registerizer_internal::registerized_store<T, T (&)[N], N>(arr, i, (std::remove_reference_t<T>) val);
        return val;
    }

    template<typename T, int N>
    [[nodiscard]] static inline constexpr T runtime_index_wrapper(const T (&arr)[N], const uint &i) {
        return registerizer_internal::registerized_read<T, const T (&)[N], N>(arr, i);
    }

    template<typename T, int N>
    [[nodiscard]] static inline constexpr T runtime_index_wrapper_log(const T (&arr)[N], const uint &i) {
        return registerizer_internal::registerized_dicochotomic_read<T, const T(&)[N], N>(arr, i);
    }


/**
 * STD::ARRAY
 */
    template<typename T, size_t N, typename U>
    static inline constexpr U runtime_index_wrapper(std::array<T, N> &array, const uint &i, const U &val) {
        registerizer_internal::registerized_store<T, std::array<T, N>, N>(array, i, (T) val);
        return val;
    }

    template<typename T, size_t N>
    [[nodiscard]] static inline constexpr T runtime_index_wrapper(const std::array<T, N> &array, const uint &i) {
        return registerizer_internal::registerized_read<T, std::array<T, N>, N>(array, i);
    }

    template<typename T, size_t N>
    [[nodiscard]] static inline constexpr T runtime_index_wrapper_log(const std::array<T, N> &array, const uint &i) {
        return registerizer_internal::registerized_dicochotomic_read<T, std::array<T, N>, N>(array, i);
    }

    template<typename func, typename T, size_t N>
    static inline constexpr void runtime_index_wrapper_for_all(const std::array<T, N> &array, const func &&f) {
        registerizer_internal::registerized_const_forall<func, T, std::array<T, N>, N>(array, std::forward<const func>(f));
    }

    template<typename func, typename T, size_t N>
    static inline constexpr void runtime_index_wrapper_transform_ith(std::array<T, N> &array, const uint &idx, const func &&f) {
        registerizer_internal::registerize_transform_ith<func, T, std::array<T, N>, N>(array, std::forward<const func>(f), idx);
    }


/**
 * SYCL VEC
 */
    template<template<typename, int> class vec_t, typename T, int N, typename U>
    static inline constexpr U runtime_index_wrapper(vec_t<T, N> &vec, const uint &i, const U &val) {
        registerizer_internal::registerized_store<T, vec_t<T, N>, N>(vec, i, (T) val);
        return val;
    }

    template<template<typename, int> class vec_t, typename T, int N>
    [[nodiscard]] static inline constexpr T runtime_index_wrapper(const vec_t<T, N> &vec, const uint &i) {
        return registerizer_internal::registerized_read<T, vec_t<T, N>, N>(vec, i);
    }

    template<template<typename, int> class vec_t, typename T, int N>
    [[nodiscard]] static inline constexpr T runtime_index_wrapper_log(const vec_t<T, N> &vec, const uint &i) {
        return registerizer_internal::registerized_dicochotomic_read<T, vec_t<T, N>, N>(vec, i);
    }

/**
 * SYCL ID
 */
    template<template<int> class vec_t, int N, typename U>
    static inline constexpr U runtime_index_wrapper(vec_t<N> &vec, const uint &i, const U &val) {
        registerizer_internal::registerized_store<size_t, vec_t<N>, N>(vec, i, (uint) val);
        return val;
    }

    template<template<int> class vec_t, int N>
    [[nodiscard]] static inline constexpr size_t runtime_index_wrapper(const vec_t<N> &vec, const uint &i) {
        return registerizer_internal::registerized_read<size_t, vec_t<N>, N>(vec, i);
    }


    template<template<int> class vec_t, int N>
    [[nodiscard]]  static inline constexpr size_t runtime_index_wrapper_log(const vec_t<N> &vec, const uint &i) {
        return registerizer_internal::registerized_dicochotomic_read<size_t, vec_t<N>, N>(vec, i);
    }


    /**
     * Constructs an accessor that can be used with dynnamic indices at runtime
     */
    template<class array_t>
    class runtime_wrapper {
    private:
        array_t &array_ref_;
    public:
        [[nodiscard]] explicit runtime_wrapper(array_t &arr) : array_ref_(arr) {}

        /**
         * Reading method for arrays/types with deduced size
         * @param i Index where to read
         * @return Value read
         */
        [[nodiscard]] auto read(uint i) const {
            return runtime_index_wrapper(array_ref_, i);
        }

        [[nodiscard]] auto operator[](uint i) const {
            return read(i);
        }

        /**
         * Writing method for arrays/types with deduced size
         * @tparam U Type of the value to write
         * @param i Index where to write
         * @param val Value to write
         * @return Value we have written
         */
        template<typename U>
        U write(uint i, const U &val) {
            return runtime_index_wrapper(array_ref_, i, val);
        }


        /**
         * Reading method for types with subscript that we don't know the maximum length
         * @tparam N Maximum value used of the index i
         * @param i Index where to read
         * @return Value read
         */
        template<int N>
        [[nodiscard]] auto read(uint i) const {
            return runtime_index_wrapper<N>(array_ref_, i);
        }

        template<int N>
        [[nodiscard]] auto operator[](uint i) const {
            return read<N>(i);
        }


        /**
         * Writing method for types with subscript that we don't know the maximum length
         * @tparam N Maximum value used of the index i
         * @tparam U Type of the value to store
         * @param i Index where to write
         * @param val Value to write
         * @return Value we have written
         */
        template<int N, typename U>
        U write(uint i, const U &val) {
            return runtime_index_wrapper<N>(array_ref_, i, val);
        }
    };


}