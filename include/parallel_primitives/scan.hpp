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


#include "internal/common.h"
#include "../usm_smart_ptr.hpp"
#include <numeric>

using namespace usm_smart_ptr;

namespace parallel_primitives {
    namespace internal {

        template<scan_type t, typename T, typename func>
        struct scan_kernel_prescan;

        template<scan_type t, typename T, typename func>
        struct scan_kernel_propagate;

        template<scan_type type, typename func, typename T>
        static inline void scan_device_impl(sycl::queue &q, const T *d_in, T *d_out, index_t length, sycl::nd_range<1> kernel_range) {
            const size_t group_count = kernel_range.get_group_range().size();

            q.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<scan_kernel_prescan<type, func, T>>(
                        kernel_range,
                        [length, d_in, d_out](sycl::nd_item<1> item) {
                            const func op{};
                            const size_t group_id = item.get_group_linear_id();
                            const size_t group_count = item.get_group_range().size();
                            const size_t group_global_offset = get_cumulative_work_size(group_count, group_id, length);
                            const size_t this_work_size = get_group_work_size(group_count, group_id, length);
                            const T *group_in = d_in + group_global_offset;
                            T *group_out = d_out + group_global_offset;
                            // First pass: scans
                            if (group_global_offset + this_work_size <= length) {
                                if constexpr(type == scan_type::inclusive) {
                                    sycl::joint_inclusive_scan(item.get_group(), group_in, group_in + this_work_size, group_out, op, get_init<T, func>());
                                } else if constexpr (type == scan_type::exclusive) {
                                    sycl::joint_exclusive_scan(item.get_group(), group_in, group_in + this_work_size, group_out, get_init<T, func>(), op);
                                } else {
                                    fail_to_compile<type, T, func>();
                                }
                            }
                        });
            }).wait();

            const func op{};
            if (group_count == 1) {
                return;
            }

            std::vector<T> partial_scans(group_count, get_init<T, func>());
            for (size_t c = 1; c <= group_count; c++) {
                partial_scans[c] = op(partial_scans[c - 1], d_out[get_cumulative_work_size(group_count, c, length) - 1]);
            }
            sycl::buffer<T> partial_scans_b(partial_scans.data(), sycl::range(group_count));

            q.submit([&](sycl::handler &cgh) {
                auto acc = sycl::accessor<T, 1, sycl::access_mode::read, sycl::access::target::constant_buffer>(partial_scans_b, cgh);
                cgh.parallel_for<scan_kernel_propagate<type, func, T>>(
                        kernel_range,
                        [length, d_in, d_out, acc](sycl::nd_item<1> item) {
                            const func op{};
                            const size_t group_id = item.get_group_linear_id();
                            if (group_id == 0) return;
                            const size_t item_local_offset = item.get_local_linear_id();
                            const size_t group_count = item.get_group_range().size();
                            const size_t group_size = item.get_local_range().size();
                            const size_t group_global_offset = get_cumulative_work_size(group_count, group_id, length);
                            const size_t this_work_size = get_group_work_size(group_count, group_id, length);

                            const T *group_in = d_in + group_global_offset;
                            T *group_out = d_out + group_global_offset;

                            T prev = acc[group_id];

                            if (group_global_offset + this_work_size <= length) {
                                for (size_t i = item_local_offset; i < this_work_size; i += group_size) {
                                    group_out[i] = op(group_out[i], prev);
                                }
                            }
                        });
            }).wait();
        }
    }


    template<scan_type type, typename func, typename T>
    void scan_device(sycl::queue &q, const T *input, T *output, index_t length) {

        auto max_kernel_items = std::min(
                internal::get_max_work_items<internal::scan_kernel_propagate<type, func, T>>(q),
                internal::get_max_work_items<internal::scan_kernel_prescan<type, func, T>>(q)
        );

        index_t max_items = std::min(4096ul, std::max(1ul, max_kernel_items)); // No more than 4096 items per reduction WG in DPC++

        index_t sm_count = (uint32_t) q.get_device().get_info<sycl::info::device::max_compute_units>();
        index_t work_ratio_per_item = 1024;
        max_items = std::min(max_items, length);
        sm_count = std::min(sm_count, (length + (work_ratio_per_item * max_items) - 1) / (work_ratio_per_item * max_items));
        sycl::nd_range<1> kernel_parameters(max_items * sm_count, max_items);
        internal::scan_device_impl<type, func>(q, input, output, length, kernel_parameters);
    }

    template<scan_type type, typename func, typename T>
    void group_scan_device(sycl::queue &q, const T *input, T *output, index_t length) {
        sycl::range<1> work_items = q.get_device().get_info<sycl::info::device::max_work_group_size>();
        sycl::nd_range<1> kernel_parameters = sycl::nd_range(work_items, work_items);
        internal::scan_device_impl<type, func>(q, input, output, length, kernel_parameters);
    }


    template<scan_type type, typename func, typename T>
    void scan(sycl::queue &q, const T *input, T *output, index_t length) {
        auto d_out = usm_unique_ptr<T, alloc::device>(length, q);
        auto d_in = usm_unique_ptr<T, alloc::device>(length, q);
        q.memcpy(d_in.get(), input, d_in.size_bytes()).wait();
        scan_device<type, func>(q, d_in.get(), d_out.get(), length);
        q.memcpy(output, d_out.get(), d_out.size_bytes()).wait();
    }

}