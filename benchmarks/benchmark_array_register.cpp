#include <runtime_index_wrapper.hpp>
#include <benchmark/benchmark.h>

using sycl::ext::runtime_index_wrapper;
struct my_struct {
    uint i = 1, j = 2;
    uint array[2] = {0};
    std::array<size_t, 2> some_coordinates{0, 0}, more{1, 0};
    sycl::ulong2 even_more{0, 3};
};


struct local_mem_benchmark_array;
struct local_mem_benchmark_array_register;
struct local_mem_benchmark_array_register_with_class;

size_t benchmark_array_regular(size_t size) {
    sycl::queue q{sycl::gpu_selector{}};
    volatile uint *ptr = sycl::malloc_device<uint>(1, q);
    q.parallel_for<local_mem_benchmark_array>(size, [=](sycl::id<1> id) {
        my_struct data{};
        data.array[0] = *ptr;
        uint init = *ptr;
        for (int c = 0; c < 100; ++c) {
            data.some_coordinates[0] = data.array[(c + data.array[0]) % 2];
            data.even_more[1] = data.some_coordinates[c % 2];
            data.array[(init + c) % 2] = c;
            data.more[(c + init) % 2] = data.array[1];
        }
        *ptr = data.even_more[*ptr % 2];
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
            data.some_coordinates[0] = runtime_index_wrapper(data.array, (c + data.array[0]) % 2);
            data.even_more[1] = runtime_index_wrapper(data.some_coordinates, c % 2);
            runtime_index_wrapper(data.array, (init + c) % 2, c * init);
            runtime_index_wrapper(data.more, (c + init) % 2, data.array[1]);
        }
        *ptr = runtime_index_wrapper(data.even_more, *ptr % 2);
    }).wait();
    return size * 100;
}

size_t benchmark_array_register_with_class(size_t size) {
    sycl::queue q{sycl::gpu_selector{}};
    volatile uint *ptr = sycl::malloc_device<uint>(1, q);
    q.parallel_for<local_mem_benchmark_array_register_with_class>(size, [=](sycl::id<1> id) {
        my_struct data{};
        data.array[0] = *ptr;
        uint init = *ptr;

        sycl::ext::runtime_wrapper array(data.array);
        sycl::ext::runtime_wrapper some_coordinates(data.some_coordinates);
        sycl::ext::runtime_wrapper more(data.more);
        sycl::ext::runtime_wrapper even_more(data.even_more);

        for (int c = 0; c < 100; ++c) {
            data.some_coordinates[0] = array[(c + data.array[0]) % 2];
            data.even_more[1] = some_coordinates.read(c % 2);
            array.write((init + c) % 2, c * init);
            more.write((c + init) % 2, data.array[1]);
        }
        *ptr = even_more.read(*ptr % 2);
    }).wait();
    return size * 100;
}

void stack_array(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    size_t processed_items = 0;
    for (auto _: state) {
        processed_items += benchmark_array_regular(size);
    }
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));
}

void registerized_array(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    size_t processed_items = 0;
    for (auto _: state) {
        processed_items += benchmark_array_register(size);
    }
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));
}

void registerized_array_with_class(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    size_t processed_items = 0;
    for (auto _: state) {
        processed_items += benchmark_array_register_with_class(size);
    }
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));
}

BENCHMARK(stack_array)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(3'000'000, 1073741824);
BENCHMARK(registerized_array)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(3'000'000, 1073741824);
BENCHMARK(registerized_array_with_class)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(3'000'000, 1073741824);


BENCHMARK_MAIN();