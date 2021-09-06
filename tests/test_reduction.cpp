#include <gtest/gtest.h>
#include <parallel_primitives/reduction.hpp>

void test_reduce_device(size_t size, sycl::queue q) {
    using T = uint64_t;
    auto in = usm_unique_ptr<T, alloc::shared>(size, q);
    std::iota(in.get(), in.get() + in.size(), 0);
    T res = parallel_primitives::reduce_device<sycl::plus<>>(q, in.get_span());
    ASSERT_EQ(res, (size * (size - 1)) / 2);
}

void test_reduce_host(size_t size, sycl::queue q) {
    using T = uint64_t;
    auto in = std::vector<T>(size, T{});
    std::iota(in.begin(), in.end(), 0);
    T res = parallel_primitives::reduce<sycl::plus<>>(q, sycl::span<T>{in});
    ASSERT_EQ(res, (size * (size - 1)) / 2);
}


TEST(reduction, device) {
    for (size_t i = 1; i < 1'000'000; i *= 4) {
        test_reduce_device(i, sycl::queue{sycl::gpu_selector{}});
    }
}

TEST(reduction, host) {
    for (size_t i = 1; i < 1'000'000; i *= 4) {
        test_reduce_host(i, sycl::queue{sycl::gpu_selector{}});
    }
}