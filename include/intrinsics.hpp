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

    static inline bool is_in_mask(uint32_t mask, size_t idx) {
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
    static inline uint32_t funnelshift_l(uint32_t lo, uint32_t hi, uint32_t shift) {
        if (shift == 0) return hi; // To avoid shifting by 32
        return (hi << (uint) (shift % 31)) | (lo >> (uint) (32 - (shift % 31)));
    }

    /**
     * @see https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g125eeef4993d16dc8d679b239460fc34
     */
    static inline uint32_t funnelshift_r(uint32_t lo, uint32_t hi, uint32_t shift) {
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

    template<typename T>
    static inline std::enable_if_t<std::is_same_v<T, std::byte> || std::is_same_v<T, sycl::uchar>, uint32_t>
    upsample(const T &hi_hi, const T &hi, const T &lo, const T &lo_lo) {
        uint16_t hi_upsampled = (uint16_t(hi_hi) << 8) + hi;
        uint16_t lo_upsampled = (uint16_t(lo) << 8) + lo_lo;
        return sycl::upsample(hi_upsampled, lo_upsampled);
    }


    template<typename T>
    static inline constexpr uint8_t get_byte(const T &word, const uint &idx) {
        static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
        return (word >> (8 * idx)) & 0xFF;
    }

    template<typename T>
    static inline constexpr T set_byte(const T &word, const uint8_t &byte_in, const uint &idx) {
        static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
        T select_mask = ~(T(0xFF) << (idx * 8));
        T new_val = (T(byte_in) & 0xFF) << (idx * 8);
        return (word & select_mask) + new_val;
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
    static inline T broadcast_leader(const sycl::sub_group &sg, T val) {
        return sycl::select_from_group(sg, val, 0);
    }

}



