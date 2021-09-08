#include <runtime_index_wrapper.h>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

/**
 * Size deduced
 */
void check_stack_array() {
    constexpr int size = 10;
    size_t arr[size];
    for (int i = 0; i < size; ++i) {
        runtime_index_wrapper(arr, i) = (size_t) i;
    }

    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper(arr, i), (size_t) i);
    }

}

/**
 * Size deduced
 */
void check_std_array() {
    constexpr int size = 10;
    std::array<size_t, size> arr{};
    for (int i = 0; i < size; ++i) {
        runtime_index_wrapper(arr, i) = (size_t) i;
    }

    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper(arr, i), (size_t) i);
    }
}

/**
 * Size cannot be deduced
 */
void check_std_vector() {
    constexpr int size = 10;
    std::vector<size_t> arr(size, 0);
    for (int i = 0; i < size; ++i) {
        runtime_index_wrapper<size>(arr, i) = (size_t) i;
    }

    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper<size>(arr, i), (size_t) i);
    }
}


/**
 * Size cannot be deduced
 */
void check_sycl_id() {
    sycl::id<3> id{1, 2, 3};
    ASSERT_EQ(runtime_index_wrapper<3>(id, 0), 1);
    ASSERT_EQ(runtime_index_wrapper<3>(id, 1), 2);
    runtime_index_wrapper<3>(id, 2) = 0;
    ASSERT_EQ(runtime_index_wrapper<3>(id, 2), 0);

}


TEST(runtime_index_wrapper, stack_array) {
    check_stack_array();
}

TEST(runtime_index_wrapper, std_array) {
    check_std_array();
}

TEST(runtime_index_wrapper, std_vector) {
    check_std_vector();
}

TEST(runtime_index_wrapper, sycl_id) {
    check_sycl_id();
}