#include <parallel_primitives/scan_cooperative.hpp>
#include <parallel_primitives/scan_decoupled_lookback.hpp>
#include <benchmark/benchmark.h>

/**
 * Sum should converge to PI
 */
void basel_problem_cooperative_scan(benchmark::State &state) {
    static sycl::queue q{sycl::gpu_selector{}};
    using namespace parallel_primitives;
    auto size = static_cast<size_t>(state.range(0));
    using T = float;
    auto in = sycl::malloc_shared<T>(size, q);
    auto out = sycl::malloc_shared<T>(size, q);

    for (size_t i = 0; i < size; ++i) {
        auto idx = (double) (i + 1);
        in[i] = (T) (1. / (idx * idx));
    }

    for (auto _: state) {
        cooperative_scan_device<scan_type::inclusive, sycl::plus<>>(q, in, out, size);
    }
    state.SetBytesProcessed(static_cast<int64_t>(sizeof(T) * state.iterations() * size));
    std::stringstream str;
    str << "Result: " << std::sqrt(6 * (double) out[state.range(0) - 1]);
    state.SetLabel(str.str());
    sycl::free(in, q);
    sycl::free(out, q);
}

void basel_problem_decoupled_scan(benchmark::State &state) {
    static sycl::queue q{sycl::gpu_selector{}};
    using namespace parallel_primitives;
    auto size = static_cast<size_t>(state.range(0));
    using T = uint;
    auto in = sycl::malloc_device<T>(size, q);
    auto out = sycl::malloc_device<T>(size, q);

    q.fill(in, T(1), size).wait();

    for (auto _: state) {
        decoupled_scan_device<scan_type::inclusive, sycl::plus<>>(q, in, out, size);
    }

    state.SetBytesProcessed(static_cast<int64_t>(sizeof(T) * state.iterations() * size));
    std::stringstream str;
    T out_v;
    q.memcpy(&out_v, out + size - 1, sizeof(T)).wait();
    str << "Result: " << out_v << " expected: " << size;
    state.SetLabel(str.str());
    sycl::free(in, q);
    sycl::free(out, q);
}


void basel_problem_regular_scan(benchmark::State &state) {
    static sycl::queue q{sycl::gpu_selector{}};
    using namespace parallel_primitives;
    auto size = static_cast<size_t>(state.range(0));
    using T = float;
    auto in = sycl::malloc_shared<T>(size, q);
    auto out = sycl::malloc_shared<T>(size, q);

    for (size_t i = 0; i < size; ++i) {
        auto idx = (double) (i + 1);
        in[i] = (T) (1. / (idx * idx));
    }

    for (auto _: state) {
        scan_device<scan_type::inclusive, sycl::plus<>>(q, in, out, size);
    }


    state.SetBytesProcessed(static_cast<int64_t>(sizeof(T) * state.iterations() * size));
    std::stringstream str;
    str << "Result: " << std::sqrt(6 * (double) out[size - 1]);
    state.SetLabel(str.str());
    sycl::free(in, q);
    sycl::free(out, q);
}


BENCHMARK(basel_problem_decoupled_scan)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(1'000, 500'000'000);
BENCHMARK(basel_problem_cooperative_scan)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(1'000, 500'000'000);
BENCHMARK(basel_problem_regular_scan)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(1'000, 500'000'000);

// Run benchmark
BENCHMARK_MAIN();
