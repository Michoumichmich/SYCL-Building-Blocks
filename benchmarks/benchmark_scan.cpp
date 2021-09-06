#include <parallel_primitives/scan_cooperative.hpp>
#include <parallel_primitives/scan_decoupled_lookback.hpp>
#include <benchmark/benchmark.h>
#include <usm_smart_ptr.hpp>

using namespace parallel_primitives;

/**
 * Sum should converge to PI
 */
void basel_problem_cooperative_scan(benchmark::State &state) {
    static sycl::queue q{sycl::gpu_selector{}};
    auto size = static_cast<size_t>(state.range(0));
    using T = float;
    auto in = usm_unique_ptr<T, alloc::shared>(size, q);
    auto out = usm_unique_ptr<T, alloc::shared>(size, q);

    for (size_t i = 0; i < size; ++i) {
        auto idx = (double) (i + 1);
        in.get()[i] = (T) (1. / (idx * idx));
    }

    for (auto _: state) {
        cooperative_scan_device<scan_type::inclusive, sycl::plus<>>(q, in.get(), out.get(), size);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * in.size_bytes()));
    std::stringstream str;
    str << "Result: " << std::sqrt(6 * (double) out.get()[state.range(0) - 1]);
    state.SetLabel(str.str());
}

void basel_problem_decoupled_scan(benchmark::State &state) {
    static sycl::queue q{sycl::gpu_selector{}};
    auto size = static_cast<size_t>(state.range(0));
    using T = uint;
    auto in = usm_unique_ptr<T, alloc::device>(size, q);
    auto out = usm_unique_ptr<T, alloc::device>(size, q);

    q.fill(in.get(), T(1), in.size()).wait();

    for (auto _: state) {
        decoupled_scan_device<scan_type::inclusive, sycl::plus<>>(q, in.get(), out.get(), size);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * in.size_bytes()));
    std::stringstream str;
    T out_v;
    q.memcpy(&out_v, out.get() + size - 1, sizeof(T)).wait();
    str << "Result: " << out_v << " expected: " << size;
    state.SetLabel(str.str());
}

void basel_problem_regular_scan(benchmark::State &state) {
    static sycl::queue q{sycl::gpu_selector{}};
    auto size = static_cast<size_t>(state.range(0));
    using T = float;
    auto in = usm_unique_ptr<T, alloc::shared>(size, q);
    auto out = usm_unique_ptr<T, alloc::shared>(size, q);

    for (size_t i = 0; i < size; ++i) {
        auto idx = (double) (i + 1);
        in.get()[i] = (T) (1. / (idx * idx));
    }

    for (auto _: state) {
        scan_device<scan_type::inclusive, sycl::plus<>>(q, in.get(), out.get(), size);
    }

    state.SetBytesProcessed(static_cast<int64_t>(in.size_bytes() * state.iterations()));
    std::stringstream str;
    str << "Result: " << std::sqrt(6 * (double) out.get()[size - 1]);
    state.SetLabel(str.str());
}


BENCHMARK(basel_problem_decoupled_scan)
->
Unit(benchmark::kMillisecond)
->RangeMultiplier(2)->Range(1'000, 500'000'000);
BENCHMARK(basel_problem_cooperative_scan)
->
Unit(benchmark::kMillisecond)
->RangeMultiplier(2)->Range(1'000, 500'000'000);
BENCHMARK(basel_problem_regular_scan)
->
Unit(benchmark::kMillisecond)
->RangeMultiplier(2)->Range(1'000, 500'000'000);

// Run benchmark
BENCHMARK_MAIN();
