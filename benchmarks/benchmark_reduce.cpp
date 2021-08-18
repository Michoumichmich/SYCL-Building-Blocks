#include "../parallel_primitives/reduction.hpp"
#include <benchmark/benchmark.h>

/**
 * Sum should converge to PI
 */
void basel_problem_reduce(benchmark::State &state) {
    static sycl::queue q{sycl::gpu_selector{}};
    using namespace parallel_primitives;
    auto in = sycl::malloc_shared<half>(state.range(0), q);

    for (size_t i = 0; i < state.range(0); ++i) {
        auto idx = (double) (i + 1);
        in[i] = 1. / (idx * idx);
    }

    half res{};
    for (auto _ : state) {
        res = reduce_device<sycl::plus<>>(q, in, state.range(0));
    }
    state.SetItemsProcessed(state.range(0));
    std::stringstream str;
    str << "Result: " << std::sqrt(6 * (double) res);
    state.SetLabel(str.str());
    sycl::free(in, q);
}


BENCHMARK(basel_problem_reduce)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(1'000'000, 2'000'000'000);

// Run benchmark
BENCHMARK_MAIN();
