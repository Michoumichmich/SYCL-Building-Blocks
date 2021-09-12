#include <runtime_index_wrapper.hpp>
#include <benchmark/benchmark.h>

using sycl::ext::runtime_index_wrapper;
struct my_struct {
    uint i = 1, j = 2;
    uint array[5] = {0};
    std::array<size_t, 5> some_coordinates{0, 0, 0}, more{1, 0, 2}, even_more{0, 3, 1};
};


struct local_mem_benchmark_array;
struct local_mem_benchmark_array_register;

size_t benchmark_array_regular(size_t size) {
    sycl::queue q{sycl::gpu_selector{}};
    volatile uint *ptr = sycl::malloc_device<uint>(1, q);
    q.parallel_for<local_mem_benchmark_array>(size, [=](sycl::id<1> id) {
        my_struct data{};
        data.array[0] = *ptr;
        uint init = *ptr;
        for (int c = 0; c < 100; ++c) {
            data.some_coordinates[0] = data.array[(c + data.array[0]) % sizeof(data.array)];
            data.even_more[1] = data.some_coordinates[c % sizeof(data.some_coordinates)];
            //  data.array[(init + c) % sizeof(data.array)] = c;
            //  data.more[(c + data.even_more[1]) % sizeof(data.more)] = data.array[1];
        }
        *ptr = data.even_more[*ptr % sizeof(data.even_more)];
    }).wait();
    return size * 100;
}

size_t benchmark_array_register(size_t size) {
    sycl::queue q{sycl::gpu_selector{}};
    volatile uint *ptr = sycl::malloc_device<uint>(1, q);
    q.parallel_for<local_mem_benchmark_array_register>(size, [=](sycl::id<1> id) {
        my_struct data{};
        data.array[0] = *ptr;
        uint init = *ptr;
        for (int c = 0; c < 100; ++c) {
            data.some_coordinates[0] = runtime_index_wrapper(data.array, (c + data.array[0]) % sizeof(data.array));
            data.even_more[1] = runtime_index_wrapper(data.some_coordinates, c % sizeof(data.some_coordinates));
            //    runtime_index_wrapper(data.array, (init + c) % sizeof(data.array), c * init);
            //   runtime_index_wrapper(data.more, (c + data.even_more[1]) % sizeof(data.more), data.array[1]);
        }
        *ptr = runtime_index_wrapper(data.even_more, *ptr % sizeof(data.even_more));
    }).wait();
    return size * 100;
}

void stack_array_benchmark(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    size_t processed_items = 0;
    for (auto _: state) {
        processed_items += benchmark_array_regular(size);
    }
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));
}

void register_array_benchmark(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    size_t processed_items = 0;
    for (auto _: state) {
        processed_items += benchmark_array_register(size);
    }
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));
}

BENCHMARK(register_array_benchmark)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(3'000'000, 1073741824);
BENCHMARK(stack_array_benchmark)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(3'000'000, 1073741824);


BENCHMARK_MAIN();