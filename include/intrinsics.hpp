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

#include <sycl/sycl.hpp>

namespace internal {

    bool is_in_mask(uint32_t mask, size_t idx) {
        return ((1u << idx) & mask) == (1u << idx);
    }

}

/**
 * For NVPTX macro @see https://intel.github.io/llvm-docs/clang_doxygen/NVPTX_8cpp_source.html
 * For inline PTX @see https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
 * For PTX ISA doc @see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a
 */
namespace sycl::ext {


    using sycl::ext::intel::ctz;

    /**
     * @see https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1gf939c350eafa2f13d64e278549d3a8aa
     */
    inline uint32_t funnelshift_l(uint32_t lo, uint32_t hi, uint32_t shift) {
        if (shift == 0) return hi; // To avoid shifting by 32
        return (hi << (uint) (shift % 31)) | (lo >> (uint) (32 - (shift % 31)));
    }

    /**
     * @see https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g125eeef4993d16dc8d679b239460fc34
     */
    inline uint32_t funnelshift_r(uint32_t lo, uint32_t hi, uint32_t shift) {
        if (shift == 0) return lo; // To avoid shifting by 32
        return (lo >> shift % 31) | (hi << (32 - (shift % 31)));
    }


    /**
     * Prefetches the data to the L1 cache on the CUDA Back-end, else is a no-op.
     * @see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu
     * @tparam T pointed type
     * @param ptr address to prefetch
     */
    template<typename T>
    void static inline prefetch(const T *ptr) {
#if defined (__NVPTX__) && defined(__SYCL_DEVICE_ONLY__)
        if constexpr (sizeof(ptr) == 8) {
            asm("prefetch.L1 [%0];" :  : "l"(ptr));
        } else {
            asm("prefetch.L1 [%0];" :  : "r"(ptr));
        }
#else
        (void) ptr;
#endif
    }

    template<typename T>
    void static inline prefetch_constant(const T *ptr) {
#if defined (__NVPTX__) && defined(__SYCL_DEVICE_ONLY__)
        if constexpr (sizeof(ptr) == 8) {
            asm("prefetchu.L1 [%0];" :  : "l"(ptr));
        } else {
            asm("prefetchu.L1 [%0];" :  : "r"(ptr));
        }
#else
        (void) ptr;
#endif
    }


    /**
     * Bit reversal
     */
    static inline uint32_t brev32(uint32_t num) {
        auto reverse_num = uint32_t(0);
#if defined (__NVPTX__) && defined(__SYCL_DEVICE_ONLY__)
        asm("brev.b32 %0, %1;" : "=r"(reverse_num) : "r"(num));
#else
        const int size = sizeof(uint32_t) * 8;
        for (int i = 0; i < size; i++) {
            reverse_num |= uint32_t((num & (uint32_t(1) << i)) != 0) << ((size - 1) - i);
        }
#endif
        return reverse_num;
    }

    /**
     * Bit reversal
     */
    static inline uint64_t brev64(uint64_t num) {
        auto reverse_num = uint64_t(0);
#if defined (__NVPTX__) && defined(__SYCL_DEVICE_ONLY__)
        asm("brev.b64 %0, %1;" : "=l"(reverse_num) : "l"(num));
#else
        const int size = sizeof(uint64_t) * 8;
        for (int i = 0; i < size; i++) {
            reverse_num |= uint64_t((num & (uint64_t(1) << i)) != 0) << ((size - 1) - i);
        }
#endif
        return reverse_num;
    }


    template<typename func>
    static inline uint32_t predicate_to_mask(const sycl::sub_group &sg, func &&predicate) {
        uint32_t group_count = sg.get_local_range().size();
        uint32_t out = 0;
        for (size_t gr = 0; gr < group_count; ++gr) {
            out |= uint32_t(predicate(gr) ? 1 : 0) << gr;
        }
        return out;
    }

    static inline uint32_t ballot(const sycl::sub_group &sg, int predicate) {
        uint32_t local_val = (predicate ? 1u : 0u) << (sg.get_local_linear_id());
        return sycl::reduce_over_group(sg, local_val, sycl::plus<>());
    }


    template<typename T>
    static inline uint32_t match_any(const sycl::sub_group &sg, T val) {
        size_t local_range = sg.get_local_range().size();
        uint32_t found = 0;
        for (uint32_t i = 0; i < local_range; ++i) {
            const T from_other = sycl::select_from_group(sg, val, i);
            found |= (from_other == val ? 1u : 0u) << i;
        }
        return found;
    }


    template<typename T>
    static inline bool match_all(const sycl::sub_group &sg, uint32_t mask, T val) {
        if (mask == 0) return false;
        size_t first_work_item_id = sycl::ext::intel::ctz(mask);
        if (first_work_item_id > sg.get_local_range().size()) return false;
        const T from_others = sycl::select_from_group(sg, val, first_work_item_id);
        return mask == (ballot(sg, val == from_others) & mask);
    }


    template<typename T>
    T broadcast_leader(const sycl::sub_group &sg, T val) {
        return sycl::select_from_group(sg, val, 0);
    }

}


#define SYCL_ASSERT(x) \
if(!(x)) {volatile int * ptr = nullptr ; *ptr;}

template<bool b = false>
void check_builtins() {
    uint32_t hi = 0xDEADBEEF, lo = 0xCAFED00D;
    SYCL_ASSERT(sycl::ext::funnelshift_l(lo, hi, 8) == 0xADBEEFCA)
    SYCL_ASSERT(sycl::ext::funnelshift_l(lo, hi, 0) == 0xDEADBEEF)

    SYCL_ASSERT(sycl::ext::funnelshift_r(lo, hi, 8) == 0xEFCAFED0)
    SYCL_ASSERT(sycl::ext::funnelshift_r(lo, hi, 0) == 0xCAFED00D)

    SYCL_ASSERT(sycl::ext::brev32(2u) == 1u << 30)
    SYCL_ASSERT(sycl::ext::brev32(0xFu) == 0xFu << 28)
    SYCL_ASSERT(sycl::ext::brev32(0) == 0)
    SYCL_ASSERT(sycl::ext::brev32(sycl::ext::brev32(lo)) == lo)

    SYCL_ASSERT(sycl::ext::brev64(1) == 1ul << 63)
    SYCL_ASSERT(sycl::ext::brev64(0xFu) == 0xFul << 60)
    SYCL_ASSERT(sycl::ext::brev64(0) == 0)
    SYCL_ASSERT(sycl::ext::brev64(sycl::ext::brev64(lo)) == lo)
}

template<bool b = false>
void check_builtins(sycl::queue q) {
    bool is_host = q.is_host();
    q.single_task<class tests>([]() {
        check_builtins();
    }).wait_and_throw();

    q.parallel_for<class tests2>(sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> it) {
        check_builtins();
        if (is_host) return;
        auto sg = it.get_sub_group();
        auto mask_all = sycl::ext::predicate_to_mask(sg, [&](size_t i) { return true; }); // Select threads where tid is even
        auto mask_even = sycl::ext::predicate_to_mask(sg, [&](size_t i) { return i % 2 == 0; }); // Select threads where tid is even
        auto mask_odd = sycl::ext::predicate_to_mask(sg, [&](size_t i) { return i % 2 != 0; }); // Select threads where tid is even

        SYCL_ASSERT(sycl::popcount(sycl::ext::ballot(sg, 1)) == sg.get_local_range().size())
        SYCL_ASSERT(0 == sycl::ext::broadcast_leader(sg, sg.get_local_linear_id()))
        SYCL_ASSERT(sycl::ext::match_all(sg, 1, 1))
        SYCL_ASSERT(sycl::ext::match_all(sg, mask_all, 0xDEADBEEF))
        SYCL_ASSERT(!sycl::ext::match_all(sg, mask_all, it.get_local_linear_id()))

        SYCL_ASSERT(sycl::ext::match_all(sg, mask_even, it.get_local_linear_id() % 2))

        uint32_t expected = (it.get_local_linear_id() % 2 == 0) ? mask_even : mask_odd;
        SYCL_ASSERT(expected == sycl::ext::match_any(sg, it.get_local_linear_id() % 2 == 0))

        int val;
        sycl::ext::prefetch(&val);


    }).wait_and_throw();
}
