#include <gtest/gtest.h>
#include <intrinsics.hpp>

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

    SYCL_ASSERT(sycl::ext::upsample<unsigned char>('S', 'Y', 'C', 'L') == 0x5359434c)


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
        auto mask_all = sycl::ext::predicate_to_mask(sg, [&](size_t) { return true; }); // Select threads where tid is even
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


TEST(intrinsics, device) {
    check_builtins(sycl::queue{sycl::gpu_selector{}});
    ASSERT_TRUE(true);
}

TEST(intrinsics, host) {
    check_builtins(sycl::queue{sycl::host_selector{}});
    ASSERT_TRUE(true);
}

TEST(intrinsics, cpp) {
    check_builtins();
    ASSERT_TRUE(true);
}
