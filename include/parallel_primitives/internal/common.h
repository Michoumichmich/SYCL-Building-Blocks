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
 */

#pragma once

#include "../../utils.hpp"

namespace parallel_primitives {
    using index_t = uint64_t;

    enum class scan_type {
        inclusive,
        exclusive
    };
}

namespace parallel_primitives::internal {


    template<typename T>
    constexpr bool is_sycl_arithmetic() {
        return std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>;
    }

    template<typename T, typename func>
    constexpr static inline T get_init() {
#ifndef SYCL_IMPLEMENTATION_HIPSYCL
        static_assert(sycl::has_known_identity<func, T>::value);
        return sycl::known_identity<func, T>::value;
#else
        if constexpr(std::is_same_v<func, sycl::plus<T>> && is_sycl_arithmetic<T>()) {
            return T{};
        } else if constexpr (std::is_same_v<func, sycl::multiplies<T>> && is_sycl_arithmetic<T>()) {
            return T{1};
        } else if constexpr((std::is_same_v<func, sycl::bit_or<T>> || std::is_same_v<func, sycl::bit_xor<T>>) && std::is_unsigned_v<T>) {
            return T{};
        } else if constexpr (std::is_same_v<func, sycl::bit_and<T>> && std::is_unsigned_v<T>) {
            return ~T{};
        } else if constexpr (std::is_same_v<func, sycl::minimum<T>> && std::is_floating_point_v<T> && std::numeric_limits<T>::has_infinity) {
            return std::numeric_limits<T>::infinity(); // +INF only for floating point that has infinity
        } else if constexpr (std::is_same_v<func, sycl::minimum<T>> && !std::numeric_limits<T>::has_infinity) {
            return std::numeric_limits<T>::max();
        } else if constexpr (std::is_same_v<func, sycl::maximum<T>> && std::is_floating_point_v<T> && std::numeric_limits<T>::has_infinity) {
            return -std::numeric_limits<T>::infinity(); // -INF only for floating point that has infinity
        } else if constexpr (std::is_same_v<func, sycl::maximum<T>>) {
            return std::numeric_limits<T>::lowest();
        } else {
            fail_to_compile<T, func>();
            return 0;
        }
#endif
    }


    template<typename T, int dim>
    using local_accessor = sycl::accessor<T, dim, sycl::access_mode::read_write, sycl::access::target::local>;

    static inline size_t get_group_work_size(const size_t &group_count, const size_t &group_id, const size_t &length) {
        size_t work_per_group = length / group_count;
        size_t remainder = length % group_count;
        if (group_id < remainder) return work_per_group + 1;
        return work_per_group;
    }

    static inline size_t get_cumulative_work_size(const size_t &group_count, const size_t &group_id, const size_t &length) {
        size_t even_work_group = group_id * (length / group_count);
        size_t remainder = length % group_count;
        size_t extra_previous_work = sycl::min(group_id, remainder);
        return even_work_group + extra_previous_work;
    }

    template<typename KernelName>
    size_t get_max_work_items(sycl::queue &q) {
#ifndef SYCL_IMPLEMENTATION_HIPSYCL
        sycl::kernel_id id = sycl::get_kernel_id<KernelName>();
        auto kernel = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context()).get_kernel(id);
        return kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device());
#else
        return q.get_device().get_info<sycl::info::device::max_work_group_size>();
#endif
    }

}