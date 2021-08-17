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

namespace parallel_primitives {
    namespace internal {

        template<scan_type t, typename T, typename func>
        struct cooperative_scan_kernel;

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

        template<scan_type type, typename func, typename T>
        static inline void scan_cooperative_device(sycl::queue &q, T *d_out, const T *d_in, index_t length, sycl::nd_range<1> kernel_range) {
            auto grid_barrier = nd_range_barrier<1>::make_barrier(q, kernel_range);
            auto all_but_first_barrier = nd_range_barrier<1>::make_barrier(q, kernel_range, [](size_t i) { return i != 0; });

            q.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<cooperative_scan_kernel<type, func, T>>(
                        kernel_range,
                        [length, d_in, d_out, grid_barrier, all_but_first_barrier](sycl::nd_item<1> item) {
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
        std::chrono::time_point<std::chrono::steady_clock> start_ct1;
        std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
        start_ct1 = std::chrono::steady_clock::now();

        sycl::nd_range<1> kernel_parameters = get_max_occupancy<internal::cooperative_scan_kernel<type, func, T>>(q);
        internal::scan_cooperative_device<type, func>(q, output, input, length, kernel_parameters);

        stop_ct1 = std::chrono::steady_clock::now();
        double elapsedTime = std::chrono::duration<double, std::milli>(stop_ct1 - start_ct1).count();
        printf("Time in cooperative scan: %f \n", elapsedTime);
    }

    template<scan_type type, typename func, typename T>
    void cooperative_scan(sycl::queue &q, const T *input, T *output, index_t length) {
        auto d_out = sycl::malloc_device<T>(length, q);
        auto d_in = sycl::malloc_device<T>(length, q);

        q.memcpy(d_in, input, length * sizeof(T)).wait();

        cooperative_scan_device<type, func>(q, d_in, d_out, length);

        q.memcpy(output, d_out, length * sizeof(T)).wait();

        sycl::free(d_out, q);
        sycl::free(d_in, q);
    }

}