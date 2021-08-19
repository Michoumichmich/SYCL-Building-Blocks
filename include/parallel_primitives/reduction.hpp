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
#include <numeric>

namespace parallel_primitives {
    namespace internal {

        template<typename T, typename func>
        struct reduction_kernel;
        template<typename T, typename func>
        struct reduction_kernel2;

        template<typename func, typename T>
        static inline T reduce_device_impl(sycl::queue &q, const T *d_in, sycl::nd_range<1> kernel_range) {
            T reduced = get_init<T, func>();
            {
                sycl::buffer<T> reducedBuf(&reduced, 1);
                q.submit([&](sycl::handler &cgh) {
                    auto reduction = sycl::reduction(reducedBuf, cgh, func{});
                    cgh.parallel_for<reduction_kernel<func, T>>(
                            kernel_range, reduction,
                            [d_in](sycl::nd_item<1> item, auto &reducer) {
                                reducer.combine(d_in[item.get_global_linear_id()]);
                            });
                }).wait();
            }
            return reduced;
        }

        template<typename func, typename T>
        static inline T reduce_device_impl2(sycl::queue &q, const T *d_in, sycl::nd_range<1> kernel_range, index_t length) {
            T reduced = get_init<T, func>();
            const size_t group_count = kernel_range.get_group_range().size();
            const func op{};
            std::vector<T> partial_scans(group_count, get_init<T, func>());
            {
                sycl::buffer<T> partial_scans_b(partial_scans.data(), sycl::range(group_count));

                q.submit([&](sycl::handler &cgh) {
                    auto acc = sycl::accessor<T, 1, sycl::access_mode::write, sycl::access::target::global_buffer>(partial_scans_b, cgh);
                    sycl::stream os(1024, 256, cgh);
                    cgh.parallel_for<reduction_kernel2<func, T>>(
                            kernel_range,
                            [length, d_in, os, acc](sycl::nd_item<1> item) {
                                const func op{};
                                const size_t group_id = item.get_group_linear_id();
                                const size_t group_count = item.get_group_range().size();
                                const size_t group_global_offset = get_cumulative_work_size(group_count, group_id, length);
                                const size_t this_work_size = get_group_work_size(group_count, group_id, length);
                                const T *group_in = d_in + group_global_offset;
                                //               os << this_work_size << sycl::endl;
                                // First pass: scans
                                if (group_global_offset + this_work_size <= length) {
                                    T res = sycl::joint_reduce(item.get_group(), group_in, group_in + this_work_size, get_init<T, func>(), op);
                                    if (item.get_local_linear_id() == 0) {
                                        acc[group_id] = res;
                                    }
                                }
                            });
                }).wait();
            }


            for (size_t c = 0; c < group_count; c++) {
                reduced = op(reduced, partial_scans[c]);
            }

            return reduced;
        }
    }


    template<typename func, typename T>
    T reduce_device(sycl::queue &q, const T *input, index_t length) {
        index_t max_items = (uint32_t) std::min(4096ul, std::max(1ul, q.get_device().get_info<sycl::info::device::max_work_group_size>())); // No more than 4096 items per reduction WG in DPC++
        index_t sm_count = (uint32_t) q.get_device().get_info<sycl::info::device::max_compute_units>();
        /**
         * We cannot submit arbitrarily big ranges for kernels. We'll use the limit of INT_MAX
         */
        T out = get_init<T, func>();
        if (length < 100'000'000 || q.get_device().is_host() || q.get_device().is_cpu()) {
            const func op{};
            index_t max_kernel_global = std::numeric_limits<int32_t>::max();
            for (index_t processed = 0; processed < length;) {
                index_t chunk_size = std::min(max_kernel_global, (length - processed));
                index_t group_count = (chunk_size / max_items);
                if (chunk_size != length - processed && group_count > sm_count) { // If there will be other to process later
                    group_count -= group_count % sm_count; // submitting a number of groups proportional to the number of streaming multiprocessors/threads.
                }
                if (group_count > 0) {
                    // printf("%d\n", max_items * group_count);
                    sycl::nd_range<1> kernel_parameters(max_items * group_count, max_items);
                    out = op(out, internal::reduce_device_impl<func>(q, input + processed, kernel_parameters));
                    processed += group_count * max_items;
                } else { //group_count == 0
                    sycl::nd_range<1> kernel_parameters(chunk_size, chunk_size);
                    out = op(out, internal::reduce_device_impl<func>(q, input + processed, kernel_parameters));
                    processed += chunk_size;
                }
            }
        } else { // GPU && length > 100'000'000 works better
            index_t work_ratio_per_item = 128;
            max_items = std::min(max_items, length);
            sm_count = std::min(sm_count, (length + (work_ratio_per_item * max_items) - 1) / (work_ratio_per_item * max_items));
            sycl::nd_range<1> kernel_parameters(max_items * sm_count, max_items);
            out = internal::reduce_device_impl2<func>(q, input, kernel_parameters, length);

        }
        return out;
    }


    template<typename func, typename T>
    T reduce(sycl::queue &q, const T *input, index_t length) {
        T *d_in = sycl::malloc_device<T>(length, q);
        q.memcpy(d_in, input, length * sizeof(T)).wait();
        T out = reduce_device<func>(q, d_in, length);
        sycl::free(d_in, q);
        return out;
    }
}