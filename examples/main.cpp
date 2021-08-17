#include "../intrinsics.hpp"

#include "../parallel_primitives/scan.hpp"
#include "../parallel_primitives/scan_cooperative.hpp"

/**
 * Sum should converge to PI
 */
double basel_problem_pi(sycl::queue &q) {
    using namespace parallel_primitives;

    constexpr size_t arr_size = 1'000'000;
    std::vector<double> in(arr_size);
    std::vector<double> out(arr_size);

    for (size_t i = 0; i < arr_size; ++i) {
        auto idx = (double) (i + 1);
        in[i] = 1. / (idx * idx);
    }

    scan<scan_type::inclusive, sycl::plus<>>(q, in.data(), out.data(), arr_size);
    for (size_t i = 0; i < arr_size; i += 200'000) {
        printf("%1.16f \n", std::sqrt(6 * out[i]));
    }

    cooperative_scan<scan_type::inclusive, sycl::plus<>>(q, in.data(), out.data(), arr_size);

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

    scan<scan_type::exclusive, sycl::multiplies<>>(q, in.data(), out.data(), arr_size);
    for (size_t i = 0; i < arr_size; i += 4'000'000) {
        printf("%1.16f \n", 2. * (double) out[i]);
    }

    cooperative_scan<scan_type::exclusive, sycl::multiplies<>>(q, in.data(), out.data(), arr_size);
    for (size_t i = 0; i < arr_size; i += 4'000'000) {
        printf("%1.16f \n", 2. * (double) out[i]);
    }

    return out[arr_size - 1];
}


int main() {
    sycl::queue q{sycl::gpu_selector{}};
    basel_problem_pi(q);
    wallis_product_pi(q);

    check_builtins();
    check_builtins(q);
    check_builtins(sycl::queue{sycl::host_selector{}});

}