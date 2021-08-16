#pragma once

#include <sycl/sycl.hpp>

namespace parallel_primitives {

    template<class ... args>
    struct nothing_matched : std::false_type {
    };

    template<class ... args>
    constexpr void fail_to_compile() {
        static_assert(nothing_matched<args...>::value);
    }


    template<typename T, typename func>
    constexpr static inline T get_init() {
        if constexpr(std::is_same_v<func, sycl::plus<>> && std::is_arithmetic_v<T>) {
            return T{};
        } else if constexpr (std::is_same_v<func, sycl::multiplies<>>) {
            return T{1};
        } else if constexpr((std::is_same_v<func, sycl::bit_or<>> || std::is_same_v<func, sycl::bit_xor<>>) && std::is_unsigned_v<T>) {
            return T{};
        } else if constexpr (std::is_same_v<func, sycl::bit_and<>> && std::is_unsigned_v<T>) {
            return ~T{};
        } else if constexpr (std::is_same_v<func, sycl::minimum<>> && std::is_floating_point_v<T> && std::numeric_limits<T>::has_infinity()) {
            return std::numeric_limits<T>::infinity(); // +INF only for floating point that has infinity
        } else if constexpr (std::is_same_v<func, sycl::minimum<>> && !std::numeric_limits<T>::has_infinity()) {
            return std::numeric_limits<T>::max();
        } else if constexpr (std::is_same_v<func, sycl::maximum<>> && std::is_floating_point_v<T> && std::numeric_limits<T>::has_infinity()) {
            return -std::numeric_limits<T>::infinity(); // -INF only for floating point that has infinity
        } else if constexpr (std::is_same_v<func, sycl::maximum<>>) {
            return std::numeric_limits<T>::lowest();
        } else {
            fail_to_compile<T, func>();
            return 0;
        }
    }

    enum class scan_type {
        inclusive,
        exclusive
    };

    template<typename T, int dim>
    using local_accessor = sycl::accessor<T, dim, sycl::access_mode::read_write, sycl::access::target::local>;
}