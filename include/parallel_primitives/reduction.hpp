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

        template<typename T, typename func>
        struct reduction_kernel;

        template<typename func, typename T>
        static inline T reduce_device_impl(sycl::queue &q, const T *d_in, index_t length, sycl::nd_range<1> kernel_range) {
            T reduced = get_init<T, func>();
            {
                sycl::buffer<T> reducedBuf(&reduced, 1);
                q.submit([&](sycl::handler &cgh) {
                    auto reduction = sycl::reduction(reducedBuf, cgh, func{});
                    cgh.parallel_for<reduction_kernel<func, T>>(
                            kernel_range, reduction,
                            [length, d_in](sycl::nd_item<1> item, auto &reducer) {
                                reducer.combine(d_in[item.get_global_linear_id()]);
                            });
                }).wait();
            }
            return reduced;
        }
    }

    template<typename func, typename T>
    T reduce_device(sycl::queue &q, const T *input, index_t length) {
        size_t max_items = (uint32_t) std::max(1ul, q.get_device().get_info<sycl::info::device::max_work_group_size>() / 2);
        size_t max_groups = (uint32_t) q.get_device().get_info<sycl::info::device::max_compute_units>();
        sycl::nd_range<1> kernel_parameters(max_items * max_groups, max_items);
        return internal::reduce_device_impl<func>(q, input, length, kernel_parameters);
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