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
#include "scan.hpp"
#include "../cooperative_groups.hpp"
#include "internal/partition_descriptor.h"

namespace parallel_primitives {
    namespace internal {
        template<typename T, typename func>
        //using partition_descriptor = decoupled_lookback_internal::partition_descriptor_impl<T, func, (sizeof(decoupled_lookback_internal::data<T, func>) <= 8)>;
        using partition_descriptor = decoupled_lookback_internal::partition_descriptor_impl<T, func, false>;

        template<scan_type type, typename T, typename func>
        static inline T scan_over_group(const sycl::nd_item<1> &item, const size_t &length, const T *in, T *out, const T init = get_init<T, func>()) {
            if constexpr(type == scan_type::inclusive) {
                sycl::joint_inclusive_scan(item.get_group(), in, in + length, out, func{}, init);
            } else if constexpr (type == scan_type::exclusive) {
                sycl::joint_exclusive_scan(item.get_group(), in, in + length, out, init, func{});
            } else {
                fail_to_compile<type, T, func>();
            }
            //item.barrier();
            return out[length - 1];
        }

        template<scan_type type, typename T, typename func>
        static inline void scan_over_sub_group(const sycl::nd_item<1> &item, const size_t &length, T *inout, const size_t &thread_id, const size_t &thread_count) {
            const func op{};
            for (size_t i = thread_id; i < length; i += thread_count) {
                if constexpr(type == scan_type::inclusive) {
                    inout[i] = sycl::inclusive_scan_over_group(item.get_sub_group(), inout[i], op);
                } else if constexpr (type == scan_type::exclusive) {
                    inout[i] = sycl::exclusive_scan_over_group(item.get_sub_group(), inout[i], op);
                } else {
                    fail_to_compile<type, T, func>();
                }
            }
        }


        template<typename T>
        static inline void load_local(const T *in, const size_t &length, T *acc, const size_t &thread_id, const size_t &thread_count) {
            for (size_t i = thread_id; i < length; i += thread_count) {
                acc[i] = in[i];
            }
        }

        template<typename T, typename func>
        static inline T load_local_and_reduce(const sycl::nd_item<1> &item, const T *in, const size_t &length, T *acc, const size_t &thread_id, const size_t &thread_count) {
            const func op{};
            T reduced = get_init<T, func>();
            for (size_t i = thread_id; i < length; i += thread_count) {
                T tmp = in[i];
                acc[i] = tmp;
                reduced = op(reduced, tmp);
            }
            return reduced;//sycl::reduce_over_group(item.get_group(), reduced, op);
        }

        template<typename T, typename func>
        static inline void store_to_global_and_increment(T *out, const size_t &length, T *acc, const size_t &thread_id, const size_t &thread_count, const T &init) {
            const func op{};
            for (size_t i = thread_id; i < length; i += thread_count) {
                out[i] = op(acc[i], init);
            }
        }


        template<scan_type t, typename T, typename func>
        struct decoupled_scan_kernel;

        template<scan_type type, typename func, typename T>
        static inline void scan_decoupled_device(sycl::queue &q, const T *d_in, T *d_out, index_t length, sycl::nd_range<1> kernel_range) {
            size_t local_mem_length = q.get_device().get_info<sycl::info::device::local_mem_size>() / sizeof(T);
            //   std::cout << local_mem_length << std::endl;
            const size_t group_size = kernel_range.get_local_range().size();
            local_mem_length -= group_size; // Correction for DPC++
            local_mem_length = group_size * (local_mem_length / group_size);

            const size_t partition_count = (length + local_mem_length - 1) / local_mem_length;
            auto partitions = sycl::malloc_device<partition_descriptor<T, func>>(partition_count, q);
            sycl::event init = q.fill(partitions, partition_descriptor<T, func>{}, partition_count);

            q.submit([&](sycl::handler &cgh) {
                sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local> shared_mem(sycl::range<1>(local_mem_length), cgh);
                sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local> shared_prefix(sycl::range<1>(1), cgh);
                sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> shared_ready_state(sycl::range<1>(1), cgh);
                cgh.depends_on(init);
                cgh.parallel_for<decoupled_scan_kernel<type, func, T >>(
                        kernel_range,
                        [length, d_in, d_out, local_mem_length, shared_mem, partitions, shared_ready_state, shared_prefix](sycl::nd_item<1> item) {
                            const size_t group_id = item.get_group_linear_id();
                            const size_t thread_id = item.get_local_linear_id();
                            const size_t group_count = item.get_group_range().size();
                            const size_t group_size = item.get_local_range().size();
                            const func op{};
                            T *shared = shared_mem.get_pointer();

                            for (size_t partition_id = group_id; partition_id * local_mem_length < length; partition_id += group_count) {
                                const T *group_in = d_in + partition_id * local_mem_length;
                                T *group_out = d_out + partition_id * local_mem_length;
                                size_t this_chunk_length = sycl::min(local_mem_length, length - partition_id * local_mem_length);
                                auto partition = partitions + partition_id;

                                if (thread_id == 0) {
                                    auto res = partition_descriptor<T, func>::is_ready(partitions, partition_id);
                                    if (res) {
                                        shared_ready_state[0] = true;
                                        shared_prefix[0] = *res;
                                    } else {
                                        shared_ready_state[0] = false;
                                    }
                                }
                                item.barrier(sycl::access::fence_space::local_space);

                                if (shared_ready_state[0] == true) {
                                    T aggregate = load_local_and_reduce<T, func>(item, group_in, this_chunk_length, shared, thread_id, group_size);
                                    if (thread_id == 0) {
                                        partition->set_prefix(op(aggregate, shared_prefix[0]));
                                    }
                                    scan_over_group<type, T, func>(item, this_chunk_length, shared, group_out, shared_prefix[0]);
                                    //scan_over_sub_group<type, T, func>(item, this_chunk_length, shared, thread_id, group_size);
                                } else {
                                    T aggregate = scan_over_group<type, T, func>(item, this_chunk_length, group_in, shared);
                                    //load_local<T>(group_in, this_chunk_length, shared, thread_id, group_size);
                                    //scan_over_sub_group<type, T, func>(item, this_chunk_length, shared, thread_id, group_size);
                                    if (thread_id == 0) {
                                        partition->set_aggregate(aggregate);
                                        shared_prefix[0] = partition_descriptor<T, func>::run_look_back(partitions, partition_id);
                                        partition->set_prefix(op(aggregate, shared_prefix[0]));
                                    }
                                    item.barrier(sycl::access::fence_space::local_space);
                                    store_to_global_and_increment<T, func>(group_out, this_chunk_length, shared, thread_id, group_size, shared_prefix[0]);
                                }
                            }
                        });
            }).wait();
            sycl::free(partitions, q);
        }
    }

    template<scan_type type, typename func, typename T>
    static inline void host_scan(const T *input, T *output, index_t length) {
        const func op{};
        if (length == 0) {
            return;
        }
        if constexpr(type == scan_type::inclusive) {
            output[0] = input[0];
            for (index_t i = 1; i < length; ++i) {
                output[i] = op(input[i], output[i - 1]);
            }
        } else if constexpr (type == scan_type::exclusive) {
            output[0] = get_init<T, func>();
            for (index_t i = 1; i < length; ++i) {
                output[i] = op(input[i - 1], output[i - 1]);
            }
        } else {
            fail_to_compile<type, T, func>();
        }
    }


    template<scan_type type, typename func, typename T>
    void decoupled_scan_device(sycl::queue &q, const T *input, T *output, index_t length) {
        if (length < 65536 && q.get_device().is_gpu()) {
            return scan_device<type, func, T>(q, input, output, length);
        }

        sycl::nd_range<1> kernel_parameters = get_max_occupancy<internal::decoupled_scan_kernel<type, func, T>>(q);
        internal::scan_decoupled_device<type, func>(q, input, output, length, kernel_parameters);
    }

    template<scan_type type, typename func, typename T, bool optimised_offload = true, size_t offload_threshold = 131072>
    void decoupled_scan(sycl::queue &q, const T *input, T *output, index_t length) {
        if (optimised_offload && length < offload_threshold && q.get_device().is_gpu()) {
            host_scan<type, func, T>(input, output, length);
            return;
        }
        auto d_out = sycl::malloc_device<T>(length, q);
        auto d_in = sycl::malloc_device<T>(length, q);
        q.memcpy(d_in, input, length * sizeof(T)).wait();
        decoupled_scan_device<type, func, T, false>(q, d_in, d_out, length);
        q.memcpy(output, d_out, length * sizeof(T)).wait();
        sycl::free(d_out, q);
        sycl::free(d_in, q);
    }

}