/**
 * Array of bits accessible with runtimes indices and that is stored using larger types to reduce register look-up lacencies
 *
 * See std::bitset interface
 */

#pragma once

#include <runtime_index_wrapper.hpp>

using sycl::ext::assume;

template<int N, typename storage_type = uint32_t>
class register_bit_array {
    static_assert(std::is_unsigned_v<storage_type> && std::is_integral_v<storage_type>);

public:

    /**
     * Empty constructor, fills the array with false.
     */
    constexpr register_bit_array() noexcept {
        reset();
    }

    /**
     * Connstructor that takes a list of bytes
     * @param init
     */
    constexpr register_bit_array(const std::initializer_list<bool> &init) noexcept: register_bit_array() {
        uint idx = 0;
#pragma unroll
        for (auto b: init) {
            write(idx, b);
            ++idx;
        }
    }

    [[nodiscard]] constexpr bool test(const uint &idx) const noexcept {
        assume(idx < size());
        if constexpr(get_storage_word_count() > 64) {
            storage_type word = sycl::ext::runtime_index_wrapper_log(storage_array_, idx / word_bit_size());
            return sycl::ext::read_bit(word, idx % word_bit_size());
        } else {
            storage_type word = sycl::ext::runtime_index_wrapper(storage_array_, idx / word_bit_size());
            return sycl::ext::read_bit(word, idx % word_bit_size());
        }
    }


    [[nodiscard]] constexpr bool operator[](const uint &i) const noexcept {
        return test(i);
    }

    constexpr register_bit_array &set(const uint &idx) noexcept {
        assume(idx < size());
        sycl::ext::runtime_index_wrapper_transform_ith(
                storage_array_,
                idx / word_bit_size(), /* Word index */
                [&](const storage_type &word) {
                    return sycl::ext::set_bit_in_word<true>(word, idx % word_bit_size()); /* Position in word */
                });
        return *this;
    }

    constexpr register_bit_array &reset(const uint &idx) noexcept {
        assume(idx < size());
        sycl::ext::runtime_index_wrapper_transform_ith(
                storage_array_,
                idx / word_bit_size(),
                [&](const storage_type &word) {
                    return sycl::ext::set_bit_in_word<false>(word, idx % word_bit_size());
                });
        return *this;
    }

    constexpr register_bit_array &reset() noexcept {
#pragma unroll
        for (auto &a: storage_array_) {
            a = 0;
        }
        return *this;
    }


    constexpr register_bit_array &flip(const uint &idx) noexcept {
        assume(idx < size());
        sycl::ext::runtime_index_wrapper_transform_ith(
                storage_array_,
                idx / word_bit_size(),
                [&](const storage_type &word) {
                    return sycl::ext::flip_bit(word, idx % word_bit_size());
                });
        return *this;
    }

    constexpr register_bit_array &write(const uint &idx, bool bit) noexcept {
        assume(idx < size());
        if (bit) {
            set(idx);
        } else {
            reset(idx);
        }
        return *this;
    }

    [[nodiscard]] constexpr uint32_t count() const noexcept {
        uint32_t counter = 0;
        sycl::ext::runtime_index_wrapper_for_all(
                storage_array_,
                [&](const uint, const storage_type &word) {
                    if constexpr(std::is_same_v<storage_type, bool>) {
                        if (word) ++counter;
                    } else {
                        counter += sycl::popcount(word); // All extra bits in the storage word are set to 0
                    }
                });
        return counter;
    }

    [[nodiscard]] constexpr bool none() const noexcept {
        bool result = true;
        sycl::ext::runtime_index_wrapper_for_all(
                storage_array_,
                [&](const uint, const storage_type &word) {
                    result = result && (word == 0); // All extra bits in the storage word are set to 0
                });
        return result;
    }

    [[nodiscard]] constexpr bool any() const {
        bool result = false;
        sycl::ext::runtime_index_wrapper_for_all(
                storage_array_,
                [&](const uint, const storage_type &word) {
                    result = result || (word != 0); // All extra bits in the storage word are set to 0
                });
        return result;
    }


    [[nodiscard]] constexpr bool all() const {
        bool result = true;
        sycl::ext::runtime_index_wrapper_for_all(
                storage_array_,
                [&](const uint &i, const storage_type &word) {
                    if constexpr(std::is_same_v<storage_type, bool>) {
                        result = result && word;
                    } else {
                        if (i + 1 != get_storage_word_count()) {
                            result = result && (((word + 1) & word) == 0) && (word != 0);
                        } else {
                            result = result && ((word + 1) == (storage_type{1} << (size() % word_bit_size()))); // All extra bits in the storage word are set to 0
                        }
                    }
                });
        return result;
    }

    [[nodiscard]] constexpr int size() const noexcept {
        return N;
    }

private:

    static constexpr uint32_t word_bit_size() {
        if constexpr(std::is_same_v<storage_type, bool>) {
            return 1;
        } else {
            return sizeof(storage_type) * 8;
        }
    }

    static constexpr uint32_t get_storage_word_count() {
        return (N + word_bit_size() - 1) / word_bit_size();
    }

    std::array<storage_type, get_storage_word_count()> storage_array_{};

};


static inline void compile_time_check() {
    constexpr register_bit_array<5, bool> arr{true, false, true, false, true};
    static_assert(arr.size() == 5);
    static_assert(!arr.none());
    static_assert(arr.any());
    static_assert(!arr.all());
    static_assert(arr.count() == 3);

    constexpr register_bit_array<10, bool> arr2{true, true, true, true, true, true, true, true, true, true};
    static_assert(arr2.size() == 10);
    static_assert(!arr2.none());
    static_assert(arr2.any());
    static_assert(arr2.all());
    static_assert(arr2.count() == 10);
}
