#include "../intrinsics.hpp"

#include "../parallel_primitives/scan.hpp"
#include "../parallel_primitives/scan_cooperative.hpp"

#include <benchmark/benchmark.h>

/**
 * Sum should converge to PI
 */
void basel_problem_cooperative_scan(benchmark::State &state) {
    sycl::queue q{sycl::gpu_selector{}};
    using namespace parallel_primitives;
    constexpr size_t arr_size = 500'000'000;
    auto in = sycl::malloc_shared<float>(arr_size, q);
    auto out = sycl::malloc_shared<float>(arr_size, q);

    for (size_t i = 0; i < arr_size; ++i) {
        auto idx = (double) (i + 1);
        in[i] = 1. / (idx * idx);
    }

    for (auto _ : state) {
        cooperative_scan_device<scan_type::inclusive, sycl::plus<>>(q, in, out, state.range(0));
        state.SetBytesProcessed(sizeof(float) * state.range(0));
    }
}

void basel_problem_regular_scan(benchmark::State &state) {
    sycl::queue q{sycl::gpu_selector{}};
    using namespace parallel_primitives;
    constexpr size_t arr_size = 500'000'000;
    auto in = sycl::malloc_shared<float>(arr_size, q);
    auto out = sycl::malloc_shared<float>(arr_size, q);

    for (size_t i = 0; i < arr_size; ++i) {
        auto idx = (double) (i + 1);
        in[i] = 1. / (idx * idx);
    }

    for (auto _ : state) {
        internal::scanLargeDeviceArray<sycl::plus<>>(q, in, out, state.range(0));
        state.SetBytesProcessed(sizeof(float) * state.range(0));
    }
}


BENCHMARK(basel_problem_cooperative_scan)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(1'000'000, 500'000'000);
BENCHMARK(basel_problem_regular_scan)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(1'000'000, 500'000'000);

// Run benchmark
BENCHMARK_MAIN();
