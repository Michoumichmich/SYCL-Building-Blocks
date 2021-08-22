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

namespace sycl::ext {
    inline uint32_t funnelshift_l(uint32_t lo, uint32_t hi, uint32_t shift) {
        if (shift == 0) return hi; // To avoid shifting by 32
        return (hi << (uint) (shift % 31)) | (lo >> (uint) (32 - (shift % 31)));
    }

    inline uint32_t funnelshift_r(uint32_t lo, uint32_t hi, uint32_t shift) {
        if (shift == 0) return lo; // To avoid shifting by 32
        return (lo >> shift % 31) | (hi << (32 - (shift % 31)));
    }

    inline uint32_t brev32(uint32_t num) {
        const int size = sizeof(uint32_t) * 8;
        auto reverse_num = uint32_t(0);
        for (int i = 0; i < size; i++) {
            reverse_num |= uint32_t((num & (uint32_t(1) << i)) != 0) << ((size - 1) - i); // branchless?
            //if ((num & (1 << i)))
            //    reverse_num |= 1 << ((size - 1) - i);
        }
        return reverse_num;
    }

    inline uint64_t brev64(uint64_t num) {
        const int size = sizeof(uint64_t) * 8;
        auto reverse_num = uint64_t(0);
        for (int i = 0; i < size; i++) {
            reverse_num |= uint64_t((num & (uint64_t(1) << i)) != 0) << ((size - 1) - i); // branchless?
            //if ((num & (1ul << i)))
            //    reverse_num |= 1ul << ((size - 1) - i);
        }
        return reverse_num;
    }
}


#define SYCL_ASSERT(x) \
if(!(x)) {volatile int * ptr = nullptr ; *ptr;}

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

void check_builtins(sycl::queue q) {
    q.single_task<class tests>([]() {
        check_builtins();
    }).wait_and_throw();
}
