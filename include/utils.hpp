#pragma once

#include <sycl/sycl.hpp>

namespace sycl::ext {

    template<class ... args>
    struct false_type_tpl : std::false_type {
    };

    template<class ... args>
    constexpr void fail_to_compile() {
        static_assert(false_type_tpl<args...>::value);
    }


    template<bool B, typename true_type, typename false_type>
    struct if_t;

    template<typename true_type, typename false_type>
    struct if_t<true, true_type, false_type> {
        using type = true_type;
    };

    template<typename true_type, typename false_type>
    struct if_t<false, true_type, false_type> {
        using type = false_type;
    };


    template<typename T>
    auto constexpr get_type() {
        if constexpr(std::is_same_v<T, bool>) {
            return bool{};
        } else if constexpr (sizeof(T) == 0) {
            fail_to_compile<T>();
        } else if constexpr(sizeof(T) == 1) {
            return uint8_t{};
        } else if constexpr(sizeof(T) == 2) {
            return uint16_t{};
        } else if constexpr(sizeof(T) <= 4) {
            return uint32_t{};
        } else if constexpr(sizeof(T) <= 8) {
            return uint64_t{};
        } else {
            fail_to_compile<T>();
        }
    }


    template<typename T>
    struct smallest_storage_t {
        using type = decltype(get_type<T>());
    };

    template<typename T>
    static inline constexpr T log2(T n) {
        return ((n < 2) ? 1 : 1 + log2(n / 2));
    }

    static inline constexpr bool is_power_two(size_t n) {
        return ((n & (n - 1)) == 0);
    }
}