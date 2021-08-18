/**
 * Inspired from git@github.com:mattdean1/cuda.git
 */

#pragma once

#include "common.h"

#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

namespace parallel_primitives {
    namespace internal {
        constexpr index_t THREADS_PER_BLOCK = 512;
        constexpr index_t ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

        template<typename func, typename T, int dim>
        static void prescan_large_even(const T *input, T *output, T *reduced, const sycl::nd_item<dim> &item, sycl::accessor<T, dim, sycl::access_mode::read_write, sycl::access::target::local> local) {
            const static func op{};
            index_t blockID = item.get_group_linear_id();
            index_t threadID = item.get_local_linear_id();
            index_t elements_per_group = item.get_local_range().size() * 2;
            index_t blockOffset = blockID * elements_per_group;
            index_t ai = threadID;
            index_t bi = threadID + (elements_per_group / 2);
            index_t bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            index_t bankOffsetB = CONFLICT_FREE_OFFSET(bi);
            local[ai + bankOffsetA] = input[blockOffset + ai];
            local[bi + bankOffsetB] = input[blockOffset + bi];

            int64_t offset = 1;
            for (index_t d = elements_per_group >> 1; d > 0; d >>= 1) // build sum in place up the tree
            {
                item.barrier(sycl::access::fence_space::local_space);
                if (threadID < d) {
                    int64_t ai_bis = offset * (2 * (int64_t) threadID + 1) - 1;
                    int64_t bi_bis = offset * (2 * (int64_t) threadID + 2) - 1;
                    ai_bis += CONFLICT_FREE_OFFSET(ai_bis);
                    bi_bis += CONFLICT_FREE_OFFSET(bi_bis);
                    local[bi_bis] = op(local[bi_bis], local[ai_bis]);
                }
                offset *= 2;
            }

            item.barrier(sycl::access::fence_space::local_space);

            if (threadID == 0) {
                reduced[blockID] = local[elements_per_group - 1 + CONFLICT_FREE_OFFSET(elements_per_group - 1)];
                local[elements_per_group - 1 + CONFLICT_FREE_OFFSET(elements_per_group - 1)] = get_init<T, func>();
            }

            for (index_t d = 1; d < elements_per_group; d *= 2) // traverse down tree & build scan
            {
                offset >>= 1;

                item.barrier(sycl::access::fence_space::local_space);
                if (threadID < d) {
                    int64_t ai_bis = offset * (2 * (int64_t) threadID + 1) - 1;
                    int64_t bi_bis = offset * (2 * (int64_t) threadID + 2) - 1;
                    ai_bis += CONFLICT_FREE_OFFSET(ai_bis);
                    bi_bis += CONFLICT_FREE_OFFSET(bi_bis);

                    T t = local[ai_bis];
                    local[ai_bis] = local[bi_bis];
                    local[bi_bis] = op(local[bi_bis], t);
                }
            }

            item.barrier(sycl::access::fence_space::local_space);
            output[blockOffset + ai] = local[ai + bankOffsetA];
            output[blockOffset + bi] = local[bi + bankOffsetB];
        }

        template<typename func, typename T, int dim>
        static inline void reduce(T *output, const T *n, const sycl::nd_item<dim> &item) {
            const static func op{};
            index_t blockID = item.get_group_linear_id();
            index_t global_id = item.get_global_linear_id();
            output[global_id] = op(output[global_id], n[blockID]);
        }

        template<typename func, typename T, int dim>
        static inline void reduce(T *output, const T *n1, const T *n2, const sycl::nd_item<dim> &item) {
            const static func op{};
            index_t blockID = item.get_group_linear_id();
            index_t global_id = item.get_global_linear_id();
            output[global_id] = op(output[global_id], op(n1[blockID], n2[blockID]));
        }

        template<typename func, typename T>
        static inline void scanLargeDeviceArray(sycl::queue &q, const T *d_in, T *d_out, index_t length);

        template<typename func, typename T>
        static inline void scanSmallDeviceArray(sycl::queue &q, T *d_out, const T *d_in, index_t length) {
            q.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(length), sycl::range<1>(length)),
                        [=](sycl::nd_item<1> item) {
                            sycl::joint_exclusive_scan(item.get_sub_group(), d_in, d_in + length, d_out, get_init<T, func>(), func());
                        });
            }).wait();
        }

        template<typename func, typename T>
        static inline void scanLargeEvenDeviceArray(sycl::queue &q, const T *d_in, T *d_out, index_t length) {
            const index_t blocks = length / ELEMENTS_PER_BLOCK;
            auto d_sums = sycl::malloc_device<T>(blocks, q);
            auto d_incr = sycl::malloc_device<T>(blocks, q);

            /*
            DPCT1049:32: The workgroup size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the workgroup size if
            needed.
            */
            q.submit([&](sycl::handler &cgh) {
                local_accessor<T, 1> local_acc(sycl::range<1>(4 * THREADS_PER_BLOCK), cgh);
                cgh.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(blocks * THREADS_PER_BLOCK), sycl::range<1>(THREADS_PER_BLOCK)),
                        [=](sycl::nd_item<1> item) {
                            prescan_large_even<func>(d_in, d_out, d_sums, item, local_acc);
                        });
            }).wait();


            const index_t sumsArrThreadsNeeded = (blocks + 1) / 2;
            if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
                // perform a large scan on the sums arr
                scanLargeDeviceArray<func>(q, d_sums, d_incr, blocks);
            } else {
                // only need one block to scan sums arr so can use small scan
                scanSmallDeviceArray<func>(q, d_incr, d_sums, blocks);
            }

            q.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(blocks * ELEMENTS_PER_BLOCK), sycl::range<1>(ELEMENTS_PER_BLOCK)),
                        [=](sycl::nd_item<1> item) {
                            reduce<func>(d_out, d_incr, item);
                        });
            }).wait();

            sycl::free(d_sums, q);
            sycl::free(d_incr, q);
        }

        template<typename func, typename T>
        static inline void scanLargeDeviceArray(sycl::queue &q, const T *d_in, T *d_out, index_t length) {
            index_t remainder = length % ELEMENTS_PER_BLOCK;
            if (remainder == 0) {
                scanLargeEvenDeviceArray<func>(q, d_in, d_out, length);
            } else {
                // perform a large scan on a compatible multiple of elements
                index_t lengthMultiple = length - remainder;
                scanLargeEvenDeviceArray<func>(q, d_in, d_out, lengthMultiple);

                // scan the remaining elements and add the (inclusive) last element of the large scan to this
                T *startOfOutputArray = d_out + lengthMultiple;
                scanSmallDeviceArray<func>(q, startOfOutputArray, d_in + lengthMultiple, remainder);

                q.submit([&](sycl::handler &cgh) {
                    const T *d_in_lengthMultiple = d_in + lengthMultiple - 1;
                    T *d_out_lengthMultiple = d_out + lengthMultiple - 1;

                    cgh.parallel_for(
                            sycl::nd_range<1>(sycl::range<1>(remainder), sycl::range<1>(remainder)),
                            [=](sycl::nd_item<1> item) {
                                reduce<func>(startOfOutputArray, d_in_lengthMultiple, d_out_lengthMultiple, item);
                            });
                }).wait();
            }
        }
    }


    template<scan_type type, typename func, typename T>
    void scan(sycl::queue &q, const T *input, T *output, index_t length) {
        index_t alloc_length = length;
        index_t offset = 0;

        if constexpr (type == scan_type::inclusive) {
            alloc_length = length + 1;
            offset = 1;
        }

        auto d_out = sycl::malloc_device<T>(alloc_length, q);
        auto d_in = sycl::malloc_device<T>(alloc_length, q);
//        q.memcpy(d_out, output, length * sizeof(T)).wait();
        q.memcpy(d_in, input, length * sizeof(T)).wait();


//        std::chrono::time_point<std::chrono::steady_clock> start_ct1;
//        std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
//        start_ct1 = std::chrono::steady_clock::now();

        if (length > internal::ELEMENTS_PER_BLOCK) {
            internal::scanLargeDeviceArray<func>(q, d_in, d_out, length);
        } else {
            internal::scanSmallDeviceArray<func>(q, d_out, d_in, length);
        }

//        stop_ct1 = std::chrono::steady_clock::now();
//        double elapsedTime = std::chrono::duration<double, std::milli>(stop_ct1 - start_ct1).count();
//        printf("Time regular scan: %f \n", elapsedTime);

        q.memcpy(output, d_out + offset, length * sizeof(T)).wait();

        if (type == scan_type::inclusive && length > 1) {
            output[length - 1] = func{}(output[length - 2], input[length - 1]);
        }

        if (type == scan_type::inclusive && length == 1) {
            output[0] = input[0];
        }


        sycl::free(d_out, q);
        sycl::free(d_in, q);
    }
}



