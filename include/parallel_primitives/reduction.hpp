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
#include "../usm_smart_ptr.hpp"
#include <numeric>

using namespace usm_smart_ptr;

namespace parallel_primitives {
    namespace internal {

        template<typename T, typename func, int N>
        struct reduction_kernel;

        template<typename func, typename T, int N>
        static inline T reduce_device_impl(sycl::queue &q, const T *d_in, sycl::nd_range<1> kernel_range) {
            T reduced = get_init<T, func>();
            {
                sycl::buffer<T> reducedBuf(&reduced, 1);
                q.submit([&](sycl::handler &cgh) {
                    auto reduction = sycl::reduction(reducedBuf, cgh, func{});
                    cgh.parallel_for<reduction_kernel<func, T, N>>(
                            kernel_range, reduction,
                            [d_in](sycl::nd_item<1> item, auto &reducer) {
//                                const size_t size = item.get_sub_group().get_max_local_range().size();
//                                const size_t global_offset = N * item.get_group_linear_id() * item.get_local_range().size();
//                                const T *in = d_in + global_offset + N * size * (item.get_local_linear_id() / size) + item.get_local_linear_id() % (size);
                                const size_t size = item.get_local_range().size();
                                const T *in = d_in + N * item.get_group_linear_id() * size + item.get_local_linear_id();
#pragma unroll
                                for (size_t i = 0; i < N; ++i) {
                                    reducer.combine(in[i * size]);
                                }
                            });
                }).wait();
            }
            return reduced;
        }

    }

    template<typename func, typename T>
    static inline T host_reduce(const sycl::span<const T> &in) {
        const func op{};
        T out = get_init<T, func>();
        for (size_t i = 0; i < in.size(); ++i) {
            out = op(out, in[i]);
        }
        return out;
    }

    template<typename func, typename T, int N, int decimation_factor = 4>
    static inline T dispatch_kernel_call(sycl::queue &q, const T *input, index_t length, size_t max_items) {
        static_assert(N > 0 && N <= 256);
        static_assert(decimation_factor > 1);
        const func op{};
        T out = get_init<T, func>();
        size_t processed = 0;
        size_t scaled_length = length / N;

        if (scaled_length > 0) {
            index_t group_count = (scaled_length / max_items);
            if (group_count > 0) {
                sycl::nd_range<1> kernel_parameters(max_items * group_count, max_items);
                out = op(out, internal::reduce_device_impl<func, T, N>(q, input, kernel_parameters));
                processed += group_count * max_items * N;
            } else {
                sycl::nd_range<1> kernel_parameters(scaled_length, scaled_length);
                out = op(out, internal::reduce_device_impl<func, T, N>(q, input, kernel_parameters));
                processed += scaled_length * N;
            }
        }

        if (processed != length) {
            size_t remainder = length - processed;
            if constexpr (N > decimation_factor) {
                out = op(out, dispatch_kernel_call<func, T, N / decimation_factor>(q, input + processed, remainder, max_items));
            } else {
                out = op(out, dispatch_kernel_call<func, T, 1>(q, input + processed, remainder, max_items));
            }
            processed += remainder;
        }

        return out;
    }


    template<typename func, typename T, bool optimised_offload = true, size_t offload_threshold = 16384>
    T reduce_device(sycl::queue &q, const sycl::span<T> &input) {
        index_t max_items = (uint32_t) std::min(4096ul, std::max(1ul, q.get_device().get_info<sycl::info::device::max_work_group_size>())); // No more than 4096 items per reduction WG in DPC++
        const func op{};
        T out = get_init<T, func>();

        constexpr int unroll_size = 64;
        constexpr int decimation_factor = 16;

        /**
         * We cannot submit arbitrarily big ranges for kernels. We'll use the limit of INT_MAX
         */
        index_t max_kernel_global = std::numeric_limits<int32_t>::max();
        index_t chunk_size = 0;
        for (index_t processed = 0; processed < input.size(); processed += chunk_size) {
            chunk_size = std::min(max_kernel_global, (input.size() - processed));
            if (optimised_offload && chunk_size < offload_threshold && q.get_device().is_gpu()) {
                auto tmp = sycl::span{(T *) calloc(chunk_size, sizeof(T)), chunk_size};
                q.memcpy(tmp.data(), input.data() + processed, tmp.size_bytes()).wait();
                out = op(out, host_reduce<func, T>(tmp));
                free(tmp.data());
            } else {
                out = op(out, dispatch_kernel_call<func, T, unroll_size, decimation_factor>(q, input.data() + processed, chunk_size, max_items));
            }
        }
        return out;
    }


    template<typename func, typename T, bool optimised_offload = true, size_t offload_threshold = 16384>
    T reduce(sycl::queue &q, const sycl::span<T> &input) {
        if (input.size() < offload_threshold && optimised_offload && q.get_device().is_gpu()) {
            return host_reduce<func, T>(input);
        }
        auto d_in = usm_unique_ptr<T, alloc::device>(input.size(), q);
        q.memcpy(d_in.get(), input.data(), input.size_bytes()).wait();
        T out = reduce_device<func, T, optimised_offload, offload_threshold>(q, d_in.get_span());
        return out;
    }
}