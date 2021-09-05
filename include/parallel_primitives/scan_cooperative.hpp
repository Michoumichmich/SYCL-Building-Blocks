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

#include "common.h"
#include "../cooperative_groups.hpp"
#include <numeric>
#include "../usm_smart_ptr.hpp"

using namespace usm_smart_ptr;

namespace parallel_primitives {
    namespace internal {

        template<scan_type t, typename T, typename func>
        struct cooperative_scan_kernel;

        template<scan_type type, typename func, typename T>
        static inline void scan_cooperative_device(sycl::queue &q, const T *d_in, T *d_out, index_t length, sycl::nd_range<1> kernel_range) {
            auto grid_barrier = nd_range_barrier<1>::make_barrier(q, kernel_range);
            auto all_but_first_barrier = nd_range_barrier<1>::make_barrier(q, kernel_range, [](size_t i) { return i != 0; });

            q.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<cooperative_scan_kernel<type, func, T>>(
                        kernel_range,
                        [length2 = length, d_in, d_out, grid_barrier, all_but_first_barrier](sycl::nd_item<1> item) {
                            const size_t length = length2;
                            const func op{};
                            const size_t group_id = item.get_group_linear_id();
                            const size_t item_local_offset = item.get_local_linear_id();
                            const size_t group_count = item.get_group_range().size();
                            const size_t group_size = item.get_local_range().size();
                            const size_t group_global_offset = get_cumulative_work_size(group_count, group_id, length);
                            const size_t this_work_size = get_group_work_size(group_count, group_id, length);

                            const T *group_in = d_in + group_global_offset;
                            T *group_out = d_out + group_global_offset;

                            // First pass: inclusive scans
                            if (group_global_offset + this_work_size <= length) {
                                if constexpr(type == scan_type::inclusive) {
                                    sycl::joint_inclusive_scan(item.get_group(), group_in, group_in + this_work_size, group_out, op, get_init<T, func>());
                                } else if constexpr (type == scan_type::exclusive) {
                                    sycl::joint_exclusive_scan(item.get_group(), group_in, group_in + this_work_size, group_out, get_init<T, func>(), op);
                                } else {
                                    fail_to_compile<type, T, func>();
                                }
                            }

                            if (group_count == 1) {
                                return;
                            }

                            // Second pass: compute the intermediary sums
                            grid_barrier->wait(item);
                            T prev = get_init<T, func>();
                            if (group_global_offset + this_work_size <= length) {
                                for (size_t c = 1; c <= group_id; c++) {
                                    prev = op(prev, d_out[get_cumulative_work_size(group_count, c, length) - 1]);
                                }
                            }

                            // Third phase: propagate the results
                            all_but_first_barrier->wait(item); // The first group will never have to wait
                            if (group_global_offset + this_work_size <= length) {
                                for (size_t i = item_local_offset; i < this_work_size; i += group_size) {
                                    group_out[i] = op(group_out[i], prev);
                                }
                            }
                        });
            }).wait();
            sycl::free(grid_barrier, q);
            sycl::free(all_but_first_barrier, q);
        }
    }


    template<scan_type type, typename func, typename T>
    void cooperative_scan_device(sycl::queue &q, const T *input, T *output, index_t length) {
        sycl::nd_range<1> kernel_parameters = get_max_occupancy<internal::cooperative_scan_kernel<type, func, T>>(q);
        internal::scan_cooperative_device<type, func>(q, input, output, length, kernel_parameters);
    }


    template<scan_type type, typename func, typename T>
    void cooperative_scan(sycl::queue &q, const T *input, T *output, index_t length) {
        auto d_out = usm_unique_ptr<T, alloc::device>(length, q);
        auto d_in = usm_unique_ptr<T, alloc::device>(length, q);
        q.memcpy(d_in.get(), input, d_in.size_bytes()).wait();
        cooperative_scan_device<type, func>(q, d_in.get(), d_out.get(), d_in.size());
        q.memcpy(output, d_out.get(), d_out.size_bytes()).wait();

    }
}