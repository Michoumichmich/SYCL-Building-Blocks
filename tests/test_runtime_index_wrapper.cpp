#include <runtime_index_wrapper.hpp>
#include <gtest/gtest.h>


using sycl::ext::runtime_index_wrapper;
using sycl::ext::runtime_index_wrapper_log;
constexpr uint size = 30;

/**
 * Size deduced
 */
void check_stack_array() {
    size_t arr[size];
    for (uint i = 0; i < size; ++i) {
        runtime_index_wrapper(arr, i, i);
    }

    for (uint i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper(arr, i), (size_t) i);
    }

    const auto arr2 = arr;
    for (uint i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper<size>(arr2, i), i);
    }
}

void check_stack_array_log() {
    size_t arr[size];
    for (uint i = 0; i < size; ++i) {
        runtime_index_wrapper(arr, i, i);
    }

    for (uint i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper_log(arr, i), (size_t) i);
    }

    const auto arr2 = arr;
    for (uint i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper<size>(arr2, i), i);
    }
}

/**
 * Size deduced
 */
void check_std_array() {
    std::array<size_t, size> arr{};
    for (uint i = 0; i < size; ++i) {
        runtime_index_wrapper(arr, i, i);
    }
    for (uint i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper(arr, i), (size_t) i);
    }

    const auto arr2 = arr;
    for (uint i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper<size>(arr2, i), i);
    }
}

void check_std_array_log() {
    std::array<size_t, size> arr{};
    for (uint i = 0; i < size; ++i) {
        runtime_index_wrapper(arr, i, i);
    }
    for (uint i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper_log(arr, i), (size_t) i);
    }

    const auto arr2 = arr;
    for (uint i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper<size>(arr2, i), i);
    }
}

/**
 * Size cannot be deduced
 */
void check_std_vector() {
    std::vector<size_t> arr(size, 0);
    for (uint i = 0; i < size; ++i) {
        runtime_index_wrapper<size>(arr, i, i);
    }
    for (uint i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper<size>(arr, i), (size_t) i);
    }

    const auto arr2 = arr;
    for (uint i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper<size>(arr2, i), i);
    }
}

void check_std_vector_log() {
    std::vector<size_t> arr(size, 0);
    for (uint i = 0; i < size; ++i) {
        runtime_index_wrapper<size>(arr, i, i);
    }
    for (uint i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper_log<size>(arr, i), (size_t) i);
    }

    const auto arr2 = arr;
    for (uint i = 0; i < size; ++i) {
        ASSERT_EQ(runtime_index_wrapper<size>(arr2, i), i);
    }
}


void check_sycl_vector() {
    sycl::uint16 arr;
    for (uint i = 0; i < 16; ++i) {
        runtime_index_wrapper(arr, i, i);
    }
    for (uint i = 0; i < 16; ++i) {
        ASSERT_EQ(runtime_index_wrapper(arr, i), i);
    }

    const auto arr2 = arr;
    for (uint i = 0; i < 16; ++i) {
        ASSERT_EQ(runtime_index_wrapper(arr2, i), i);
    }

}

void check_sycl_vector_log() {
    sycl::uint16 arr;
    for (uint i = 0; i < 16; ++i) {
        runtime_index_wrapper(arr, i, i);
    }
    for (uint i = 0; i < 16; ++i) {
        ASSERT_EQ(runtime_index_wrapper_log(arr, i), i);
    }

    const auto arr2 = arr;
    for (uint i = 0; i < 16; ++i) {
        ASSERT_EQ(runtime_index_wrapper(arr2, i), i);
    }
}

void check_sycl_vector_class() {
    sycl::uint16 arr;
    sycl::ext::runtime_wrapper acc(arr);

    for (uint i = 0; i < 16; ++i) {
        acc.write(i, i);
    }
    for (uint i = 0; i < 16; ++i) {
        ASSERT_EQ(acc.read(i), i);
    }
}

void check_std_vector_class() {
    std::vector<size_t> arr(size, 0);
    sycl::ext::runtime_wrapper acc(arr);
    for (uint i = 0; i < size; ++i) {
        acc.write<size>(i, i);
    }
    for (uint i = 0; i < size; ++i) {
        ASSERT_EQ(acc.read<size>(i), i);
    }

}


/**
 * Size cannot be deduced
 */
void check_sycl_id() {
    sycl::id<3> id{1, 2, 3};
    ASSERT_EQ(runtime_index_wrapper(id, 0), 1);
    ASSERT_EQ(runtime_index_wrapper(id, 1), 2);
    runtime_index_wrapper(id, 2, 0);
    ASSERT_EQ(runtime_index_wrapper(id, 2), 0);
}

void check_sycl_id_log() {
    sycl::id<3> id{1, 2, 3};
    ASSERT_EQ(runtime_index_wrapper_log(id, 0), 1);
    ASSERT_EQ(runtime_index_wrapper_log(id, 1), 2);
    runtime_index_wrapper(id, 2, 0);
    ASSERT_EQ(runtime_index_wrapper_log(id, 2), 0);
}


TEST(runtime_index_wrapper, stack_array) {
    check_stack_array();
}

TEST(runtime_index_wrapper_log, stack_array) {
    check_stack_array_log();
}

TEST(runtime_index_wrapper, std_array) {
    check_std_array();
}

TEST(runtime_index_wrapper_log, std_array) {
    check_std_array_log();
}

TEST(runtime_index_wrapper, std_vector) {
    check_std_vector();
}

TEST(runtime_wrapper_class, std_vector) {
    check_std_vector_class();
}

TEST(runtime_index_wrapper, sycl_vector) {
    check_sycl_vector();
}

TEST(runtime_wrapper_class, sycl_vector) {
    check_sycl_vector_class();
}

TEST(runtime_index_wrapper_log, sycl_vector) {
    check_sycl_vector_log();
}

TEST(runtime_index_wrapper_log, std_vector) {
    check_std_vector_log();
}

TEST(runtime_index_wrapper, sycl_id) {
    check_sycl_id();
}

TEST(runtime_index_wrapper_log, sycl_id) {
    check_sycl_id_log();
}