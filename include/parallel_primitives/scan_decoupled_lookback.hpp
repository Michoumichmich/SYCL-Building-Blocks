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

namespace parallel_primitives {
    namespace internal {

        enum class status {
            aggregate_available,
            prefix_available,
            invalid
        };

        template<typename T, typename func>
        class partition_descriptor {
        private:
            status status_flag_ = status::invalid;
            T inclusive_prefix_ = get_init<T, func>();
            T aggregate_ = get_init<T, func>();

        public:
            inline void set_aggregate(const T &aggregate) {
                aggregate_ = aggregate;
                sycl::atomic_fence(sycl::memory_order_seq_cst, sycl::memory_scope_device);
                status_flag_ = status::aggregate_available;
            }

            inline void set_prefix(const T &prefix) {
                inclusive_prefix_ = prefix;
                sycl::atomic_fence(sycl::memory_order_seq_cst, sycl::memory_scope_device);
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

            static bool is_ready(volatile partition_descriptor *ptr_base, const size_t &partition_id) {
                if (partition_id == 0) {
                    return true;
                } else {
                    return (ptr_base[partition_id - 1].status_flag_ == status::prefix_available);
                }
            }

        };

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

        template<typename T, typename func>
        static inline T load_local_and_reduce(const sycl::nd_item<1> &item, const T *in, const size_t &length, T *acc, const size_t &thread_id, const size_t &thread_count) {
            const func op{};
            T reduced = get_init<T, func>();
            for (size_t i = thread_id; i < length; i += thread_count) {
                T tmp = in[i];
                acc[i] = tmp;
                reduced = op(reduced, tmp);
            }
            return sycl::reduce_over_group(item.get_group(), reduced, op);
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

                                if (thread_id == 0) shared_ready_state[0] = partition_descriptor<T, func>::is_ready(partitions, partition_id);
                                //     is_ready = sycl::any_of_group(item.get_group(), is_ready);
                                item.barrier();
                                if (shared_ready_state[0]) {
                                    T aggregate = load_local_and_reduce<T, func>(item, group_in, this_chunk_length, shared, thread_id, group_size);
                                    if (thread_id == 0) {
                                        partition->set_aggregate(aggregate);
                                        shared_prefix[0] = partition_descriptor<T, func>::run_look_back(partitions, partition_id);
                                        partition->set_prefix(op(aggregate, shared_prefix[0]));
                                    }
                                    item.barrier();
                                    scan_over_group<type, T, func>(item, this_chunk_length, shared, group_out, shared_prefix[0]);
                                } else {
                                    T aggregate = scan_over_group<type, T, func>(item, this_chunk_length, group_in, shared);
                                    if (thread_id == 0) {
                                        partition->set_aggregate(aggregate);
                                        shared_prefix[0] = partition_descriptor<T, func>::run_look_back(partitions, partition_id);
                                        partition->set_prefix(op(aggregate, shared_prefix[0]));
                                    }
                                    item.barrier();
                                    store_to_global_and_increment<T, func>(group_out, this_chunk_length, shared, thread_id, group_size, shared_prefix[0]);
                                }
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