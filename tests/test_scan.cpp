#include <gtest/gtest.h>
#include <parallel_primitives/scan.hpp>
#include <parallel_primitives/scan_cooperative.hpp>
#include <parallel_primitives/scan_decoupled_lookback.hpp>


/**
 * Sum should converge to PI
 */
double basel_problem_pi(sycl::queue &q) {
    using namespace parallel_primitives;

    constexpr size_t arr_size = 1'000'000;
    std::vector<float> in(arr_size);
    std::vector<float> out(arr_size);

    for (size_t i = 0; i < arr_size; ++i) {
        auto idx = (double) (i + 1);
        in[i] = 1. / (idx * idx);
    }

    scan<scan_type::inclusive, sycl::plus<>>(q, in.data(), out.data(), arr_size);
    for (size_t i = 0; i < arr_size; i += 200'000) {
        printf("%1.16f \n", std::sqrt(6 * out[i]));
    }

    decoupled_scan<scan_type::inclusive, sycl::plus<>>(q, in.data(), out.data(), arr_size);

    for (size_t i = 0; i < arr_size; i += 200'000) {
        printf("%1.16f \n", std::sqrt(6 * out[i]));
    }
    return out[arr_size - 1];
}

/**
 * Product should converge to PI
 */
float wallis_product_pi(sycl::queue &q) {
    using namespace parallel_primitives;
    constexpr int arr_size = 20'000'000;
    std::vector<float> in(arr_size);
    std::vector<float> out(arr_size);

    for (size_t i = 0; i < arr_size; ++i) {
        auto idx = (double) (i + 1);
        in[i] = (float) (4 * (idx * idx) / (4. * (idx * idx) - 1));
    }

    scan<scan_type::inclusive, sycl::multiplies<>>(q, in.data(), out.data(), arr_size);
    for (size_t i = 0; i < arr_size; i += 4'000'000) {
        printf("%1.16f \n", 2. * (double) out[i]);
    }

    decoupled_scan<scan_type::inclusive, sycl::multiplies<>>(q, in.data(), out.data(), arr_size);
    for (size_t i = 0; i < arr_size; i += 4'000'000) {
        printf("%1.16f \n", 2. * (double) out[i]);
    }

    return out[arr_size - 1];
}