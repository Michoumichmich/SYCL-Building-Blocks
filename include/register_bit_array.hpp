/**
 * Array of bits accessible with runtimes indices and that is stored using larger types to reduce register look-up lacencies
 *
 * See std::bitset interface
 */

#pragma once

#include <runtime_index_wrapper.hpp>

using sycl::ext::assume;


/**
 * @see https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
 * @tparam T
 * @param v
 * @return
 */
template<typename T>
static constexpr inline uint popcount_kerninghan(T v) {
    static_assert(std::is_unsigned_v<T> && std::is_integral_v<T>);
    uint c; // c accumulates the total bits set in v
    for (c = 0; v; c++) {
        v &= v - 1; // clear the least significant bit set
    }
    return c;
}

/**
 * For large sizes of N (>1280), the array might not fit in GPU registers, moving to uint64_t solves the issue. For smalle sizes, uint64_t is slower.
 * @tparam N Number of bits to store
 * @tparam storage_type Word type used to store the bits.
 */
template<int N, typename storage_type = uint32_t>
class register_bit_array {

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

    /**
     * Checks whether a bit is set
     * @param idx index to test
     * @return
     */
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

    /**
     * Checks whether a bit is set
     * @param idx bit index to test
     * @return
     */
    [[nodiscard]] constexpr bool operator[](const uint &i) const noexcept {
        return test(i);
    }

    /**
     * Set a bit to true
     * @param idx the position of the bit to set
     * @return *this
     */
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

    /**
     * Unsets a bit ie. sets it to false
     * @param idx the position of the bit to set
     * @return *this
     */
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

    /**
     * Sets all the bits in the to false.
     * @return *this
     */
    constexpr register_bit_array &reset() noexcept {
#pragma unroll
        for (auto &a: storage_array_) {
            a = 0;
        }
        return *this;
    }


    /**
     * Flips a bit
     * @param idx the position of the bit to flit
     * @return *this
     */
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

    /**
     * Sets a bit to val
     * @param idx the position of the bit to set
     * @param val value to store
     * @return *this
     */
    constexpr register_bit_array &write(const uint &idx, bool val) noexcept {
        assume(idx < size());
        if (val) {
            set(idx);
        } else {
            reset(idx);
        }
        return *this;
    }

    /**
     * Counts the number of bits that are set
     * @return uint32_t representing the number of bit set
     */
    [[nodiscard]] constexpr uint32_t count() const noexcept {
        uint32_t counter = 0;
        sycl::ext::runtime_index_wrapper_for_all(
                storage_array_,
                [&](const uint, const storage_type &word) {
                    if constexpr(std::is_same_v<storage_type, bool>) {
                        if (word) ++counter;
                    } else {
#ifdef SYCL_DEVICE_ONLY
                        counter += sycl::popcount(word); // All extra bits in the storage word are set to 0
#else
                        counter += popcount_kerninghan(word);
#endif
                    }
                });
        return counter;
    }

    /**
     * Checks if none of the bits are set to true
     * @return bool
     */
    [[nodiscard]] constexpr bool none() const noexcept {
        bool result = true;
        sycl::ext::runtime_index_wrapper_for_all(
                storage_array_,
                [&](const uint, const storage_type &word) {
                    result = result && (word == 0); // All extra bits in the storage word are set to 0
                });
        return result;
    }

    /**
     * Checks if any bit is set to true.
     * @return bool
     */
    [[nodiscard]] constexpr bool any() const {
        return !none();
    }


    /**
     * Checks whether all the bits are set to true
     * @return bool
     */
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
    static_assert(std::is_unsigned_v<storage_type> && std::is_integral_v<storage_type>);
};


static inline void register_bit_array_compile_time_tests() {
    constexpr register_bit_array<5, bool> arr{true, false, true, false, true};
    static_assert(arr.size() == 5);
    static_assert(!arr.none());
    static_assert(arr.any());
    static_assert(!arr.all());
    static_assert(arr.count() == 3);

    constexpr register_bit_array<10, unsigned char> arr2{true, true, true, true, true, true, true, true, true, true};
    static_assert(arr2.size() == 10);
    static_assert(!arr2.none());
    static_assert(arr2.any());
    static_assert(arr2.all());
    static_assert(arr2.count() == 10);

    constexpr register_bit_array<10, unsigned char> arr3{false, true, true, true, true, true, true, true, true, true};
    static_assert(arr3.size() == 10);
    static_assert(!arr3.none());
    static_assert(arr3.any());
    static_assert(!arr3.all());
    static_assert(arr3.count() == 9);

    constexpr register_bit_array<4, uint64_t> arr4{false, false, false, false};
    static_assert(arr4.size() == 4);
    static_assert(arr4.none());
    static_assert(!arr4.any());
    static_assert(!arr4.all());
    static_assert(arr4.count() == 0);

    constexpr int sieve_size = 100;
    constexpr auto primes_100 = [&]() {
        register_bit_array<sieve_size + 1, uint64_t> tmp{};
        for (int i = 0; i < tmp.size(); ++i)
            tmp.set(i);

        for (int p = 2; p * p <= sieve_size; p++) {
            if (tmp.test(p)) {
                for (int i = p * p; i <= sieve_size; i += p)
                    tmp.reset(i);
            }
        }

        register_bit_array<sieve_size + 1, uint8_t> primes{};
        for (int p = 2; p < tmp.size(); ++p)
            primes.write(p, tmp[p]);
        return primes;
    }();

    static_assert(!primes_100.all());
    static_assert(primes_100.any());
    static_assert(primes_100.count() == 25);
}
