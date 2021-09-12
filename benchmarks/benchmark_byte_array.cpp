#include <runtime_byte_array.hpp>
#include <benchmark/benchmark.h>


size_t benchmark_runtime_byte_array(size_t size) {
    sycl::queue q{sycl::gpu_selector{}};
    uint init = 1022;
    constexpr int array_size = 16;
    volatile uint *ptr = sycl::malloc_device<uint>(1, q);
    q.parallel_for<class runtime_byte_array_kernel>(size, [=](sycl::id<1> id) {
        runtime_byte_array<array_size> arr{static_cast<unsigned char>(*ptr), static_cast<unsigned char>(*ptr)};
        for (int c = 0; c < 100; ++c) {
            arr.write(0, arr[(c + arr[0]) % array_size]);
            arr.write(1, arr.read(c % array_size));
            arr.write((init + c) % array_size, c * init);
            arr.write((c + init) % array_size, arr[1]);
        }
        *ptr = arr.read(*ptr % array_size);
    }).wait();
    return size * 100;
}

size_t benchmark_runtime_byte_array_non_specialised(size_t size) {
    sycl::queue q{sycl::gpu_selector{}};
    uint init = 1022;
    constexpr int array_size = 16;
    volatile uint *ptr = sycl::malloc_device<uint>(1, q);
    q.parallel_for<class runtime_byte_array_non_specialised_kernel>(size, [=](sycl::id<1> id) {
        std::array<uint8_t, array_size> arr_storage{static_cast<unsigned char>(*ptr), static_cast<unsigned char>(*ptr)};
        sycl::ext::runtime_wrapper arr(arr_storage);
        for (int c = 0; c < 100; ++c) {
            arr.write(0, arr[(c + arr[0]) % array_size]);
            arr.write(1, arr.read(c % array_size));
            arr.write((init + c) % array_size, c * init);
            arr.write((c + init) % array_size, arr[1]);
        }
        *ptr = arr.read(*ptr % array_size);
    }).wait();
    return size * 100;
}

size_t benchmark_runtime_byte_array_stack(size_t size) {
    sycl::queue q{sycl::gpu_selector{}};
    uint init = 1022;
    constexpr int array_size = 16;
    volatile uint *ptr = sycl::malloc_device<uint>(1, q);
    q.parallel_for<class runtime_byte_array_stack_kernel>(size, [=](sycl::id<1> id) {
        std::array<uint8_t, array_size> arr{static_cast<unsigned char>(*ptr), static_cast<unsigned char>(*ptr)};
        for (int c = 0; c < 100; ++c) {
            arr[0] = arr[(c + arr[0]) % array_size];
            arr[1] = arr[c % array_size];
            arr[(init + c) % array_size] = c * init;
            arr[(c + init) % array_size] = arr[1];
        }
        *ptr = arr[*ptr % array_size];
    }).wait();
    return size * 100;
}

void registerized_and_optimised_byte_array(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    size_t processed_items = 0;
    for (auto _: state) {
        processed_items += benchmark_runtime_byte_array(size);
    }
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));
}

void registerized_byte_array(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    size_t processed_items = 0;
    for (auto _: state) {
        processed_items += benchmark_runtime_byte_array_non_specialised(size);
    }
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));
}


void stack_byte_array(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    size_t processed_items = 0;
    for (auto _: state) {
        processed_items += benchmark_runtime_byte_array_stack(size);
    }
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));
}


BENCHMARK(registerized_and_optimised_byte_array)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(3'000'000, 1073741824);
BENCHMARK(registerized_byte_array)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(3'000'000, 1073741824);
BENCHMARK(stack_byte_array)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(3'000'000, 1073741824);



BENCHMARK_MAIN();