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

        enum class status {
            aggregate_available,
            prefix_available,
            invalid
        };

        template<typename T, typename func>
        struct partition_descriptor {
            status status_flag_ = status::invalid;
            T aggregate_ = get_init<T, func>();
            T inclusive_prefix_ = get_init<T, func>();

            void set_aggregate(const sycl::nd_item<1> &item, const T &aggregate) {
                aggregate_ = aggregate;
                item.barrier();
                status_flag_ = status::aggregate_available;
            }

            void set_prefix(const sycl::nd_item<1> &item, const T &prefix) {
                inclusive_prefix_ = prefix;
                item.barrier();
                status_flag_ = status::prefix_available;
            }

            static T run_look_back(volatile partition_descriptor *ptr_base, const size_t &partition_id) {
                T tmp = get_init<T, func>();
                const func op{};
                for (auto partition = partition_id; partition_id > 0;) {
                    partition--;
                    while (ptr_base[partition].status_flag_ == status::invalid) {/* wait */}
                    if (ptr_base[partition].status_flag_ == status::prefix_available) {
                        return op(tmp, ptr_base[partition].inclusive_prefix_);
                    }
                    //if (ptr_base[partition].status_flag_ == status::aggregate_available) {
                    tmp = op(tmp, ptr_base[partition].aggregate_);
                    //}
                }
                return tmp;
            }
        };

        template<typename T, typename func>
        static inline T load_local_and_scan(const sycl::nd_item<1> &item, const T *in, const size_t &length, T *acc, const size_t &thread_id, const size_t &thread_count) {
            const func op{};
            for (size_t i = thread_id; i < length; i += thread_count) {
                T tmp = in[i];
                sycl::inclusive_scan_over_group(item.get_group(), tmp, op);
                //item.barrier();
                acc[i] = tmp;
            }
            item.barrier();
            return acc[length - 1];
        }

        template<typename T>
        static inline void store_to_global(const sycl::nd_item<1> &item, T *out, const size_t &length, T *acc, const size_t &thread_id, const size_t &thread_count) {
            for (size_t i = thread_id; i < length; i += thread_count) {
                out[i] = acc[i];
            }
        }

        template<typename T, typename func>
        static inline void store_to_global_and_increment(T *out, const size_t &length, T *acc, const size_t &thread_id, const size_t &thread_count, T init = get_init<T, func>()) {
            const func op{};
            for (size_t i = thread_id; i < length; i += thread_count) {
                out[i] = op(acc[i], init);
            }
        }

        template<typename T, typename func>
        static inline void increment_shared(const size_t &length, T *acc, const size_t &thread_id, const size_t &thread_count, T init) {
            const func op{};
            for (size_t i = thread_id; i < length; i += thread_count) {
                acc[i] = op(acc[i], init);
            }
        }


        template<scan_type t, typename T, typename func>
        struct decoupled_scan_kernel;

        template<scan_type type, typename func, typename T>
        static inline void scan_decoupled_device(sycl::queue &q, const T *d_in, T *d_out, index_t length, sycl::nd_range<1> kernel_range) {
            size_t local_mem_length = q.get_device().get_info<sycl::info::device::local_mem_size>() / sizeof(T);
            const size_t group_size = kernel_range.get_local_range().size();
            local_mem_length = 6 * group_size;//group_size * (local_mem_length / group_size) - 2 * group_size;

            const size_t partition_count = (length + local_mem_length - 1) / local_mem_length;
            auto partitions = sycl::malloc_device<partition_descriptor<T, func>>(partition_count, q);
            q.fill(partitions, partition_descriptor<T, func>{}, partition_count).wait();

            q.submit([&](sycl::handler &cgh) {
                sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local> acc(sycl::range<1>(local_mem_length), cgh);
                cgh.parallel_for<decoupled_scan_kernel<type, func, T>>(
                        kernel_range,
                        [length, d_in, d_out, local_mem_length, acc, partitions](sycl::nd_item<1> item) {
                            const size_t group_id = item.get_group_linear_id();
                            const size_t thread_id = item.get_local_linear_id();
                            const size_t group_count = item.get_group_range().size();
                            const size_t group_size = item.get_local_range().size();

                            for (size_t partition_id = group_id; partition_id * local_mem_length < length; partition_id += group_count) {
                                const T *group_in = d_in + partition_id * local_mem_length;
                                T *group_out = d_out + partition_id * local_mem_length;
                                size_t this_chunk_length = sycl::min(local_mem_length, length - partition_id * local_mem_length);

                                T aggregate = load_local_and_scan<T, func>(item, group_in, this_chunk_length, acc.get_pointer(), thread_id, group_size);

                                auto partition = partitions + partition_id;
                                partition->set_aggregate(item, aggregate);
                                T prefix = partition_descriptor<T, func>::run_look_back(partitions, partition_id);
                                partition->set_prefix(item, func{}(aggregate, prefix));

                                store_to_global_and_increment<T, func>(group_out, this_chunk_length, acc.get_pointer(), thread_id, group_size, prefix);
                            }

                        });
            }).wait();
            sycl::free(partitions, q);
        }
    }

    template<scan_type type, typename func, typename T>
    void decoupled_scan_device(sycl::queue &q, const T *input, T *output, index_t length) {
        sycl::nd_range<1> kernel_parameters = get_max_occupancy<internal::decoupled_scan_kernel<type, func, T>>(q);
        internal::scan_decoupled_device<type, func>(q, input, output, length, kernel_parameters);
    }

    template<scan_type type, typename func, typename T>
    void decoupled_scan(sycl::queue &q, const T *input, T *output, index_t length) {
        auto d_out = sycl::malloc_device<T>(length, q);
        auto d_in = sycl::malloc_device<T>(length, q);

        q.memcpy(d_in, input, length * sizeof(T)).wait();

        decoupled_scan_device<type, func>(q, d_in, d_out, length);

        q.memcpy(output, d_out, length * sizeof(T)).wait();

        sycl::free(d_out, q);
        sycl::free(d_in, q);
    }
}