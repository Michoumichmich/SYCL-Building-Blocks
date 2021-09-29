/**
 * Array of bits accessible with runtimes indices and that is stored using larger types to reduce register look-up latencies
 *
 * See std::bitset interface
 */

#pragma once

#include <runtime_index_wrapper.hpp>

using sycl::ext::assume;


/**
 * For large sizes of N (>1280), the array might not fit in GPU registers, moving to uint64_t solves the issue. For small sizes, uint64_t is slower.
 * @tparam N Number of bits to store
 * @tparam storage_type Word type used to store the bits.
 */
template<int N, typename storage_type = uint32_t>
class register_bit_array {

public:

    /**
     * Empty constructor, fills the array with false.
     */
    constexpr register_bit_array() noexcept;

    /**
     * Constructor that takes a list of bytes
     * @param init
     */
    constexpr register_bit_array(const std::initializer_list<bool> &init) noexcept;

    /**
     * Checks whether a bit is set
     * @param idx index to test
     * @return
     */
    [[nodiscard]] constexpr bool test(const uint &idx) const noexcept;

    /**
     * Checks whether a bit is set
     * @param idx bit index to test
     * @return
     */
    [[nodiscard]] constexpr bool operator[](const uint &i) const noexcept;

    /**
     * Set a bit to true
     * @param idx the position of the bit to set
     * @return *this
     */
    constexpr register_bit_array &set(const uint &idx) noexcept;

    /**
     * Sets all the bits in the array.
     * @return *this
     */
    constexpr register_bit_array &set() noexcept;


    /**
     * Unsets a bit ie. sets it to false
     * @param idx the position of the bit to set
     * @return *this
     */
    constexpr register_bit_array &reset(const uint &idx) noexcept;

    /**
     * Sets all the bits in the to false.
     * @return *this
     */
    constexpr register_bit_array &reset() noexcept;


    /**
     * Flips a bit
     * @param idx the position of the bit to flit
     * @return *this
     */
    constexpr register_bit_array &flip(const uint &idx) noexcept;

    /**
     * Sets a bit to val
     * @param idx the position of the bit to set
     * @param val value to store
     * @return *this
     */
    constexpr register_bit_array &write(const uint &idx, bool val) noexcept;

    /**
     * Counts the number of bits that are set
     * @return uint32_t representing the number of bit set
     */
    [[nodiscard]] constexpr uint32_t count() const noexcept;

    /**
     * Checks if any bit is set to true.
     * @return bool
     */
    [[nodiscard]] constexpr bool any() const;

    /**
     * Checks if none of the bits are set to true
     * @return bool
     */
    [[nodiscard]] constexpr bool none() const noexcept;

    /**
     * Checks whether all the bits are set to true
     * @return bool
     */
    [[nodiscard]] constexpr bool all() const;

    [[nodiscard]] constexpr int size() const noexcept { return N; }

    constexpr register_bit_array &operator|=(const register_bit_array &other);

    constexpr register_bit_array &operator^=(const register_bit_array &other);

    constexpr register_bit_array &operator&=(const register_bit_array &other);


private:

    static constexpr uint32_t word_bit_size();

    static constexpr storage_type generate_low_bit_mask(uint i);

    static constexpr uint32_t get_storage_word_count();

    /**
    * @see https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
    */
    static constexpr inline uint popcount_kerninghan(storage_type v);

    std::array<storage_type, get_storage_word_count()> storage_array_{};
    static_assert(std::is_unsigned_v<storage_type> && std::is_integral_v<storage_type>);
    static_assert(generate_low_bit_mask(10) == storage_type(0b1111111111));
};


template<int N, typename storage_type>
constexpr uint register_bit_array<N, storage_type>::popcount_kerninghan(storage_type v) {
    uint c; // c accumulates the total bits set in v
    for (c = 0U; v; c++) {
        v &= v - 1U; // clear the least significant bit set
    }
    return c;
}

template<int N, typename storage_type>
constexpr uint32_t register_bit_array<N, storage_type>::get_storage_word_count() {
    return (N + word_bit_size() - 1) / word_bit_size();
}

template<int N, typename storage_type>
constexpr storage_type register_bit_array<N, storage_type>::generate_low_bit_mask(uint i) {
    return (storage_type{1} << i) - 1;
}

template<int N, typename storage_type>
constexpr uint32_t register_bit_array<N, storage_type>::word_bit_size() {
    if constexpr(std::is_same_v<storage_type, bool>) {
        return 1;
    } else {
        return sizeof(storage_type) * 8;
    }
}

template<int N, typename storage_type>
constexpr register_bit_array<N, storage_type>::register_bit_array() noexcept {
    reset();
}

template<int N, typename storage_type>
constexpr register_bit_array<N, storage_type>::register_bit_array(const std::initializer_list<bool> &init) noexcept: register_bit_array() {
    uint idx = 0;
#pragma unroll
    for (auto b: init) {
        write(idx, b);
        ++idx;
    }
}

template<int N, typename storage_type>
constexpr bool register_bit_array<N, storage_type>::test(const uint &idx) const noexcept {
    assume(idx < size());
    /**
     * The point of this structure is to force register storage of the data so we cannot address the data.
     * Given that we're performing register lookup, for small register numbers, a linear search is the fastest
     * way, for bigger sizes, a dichotomic search performs better. (logarithmic complexity). But... that's a lot
     * of registers to go through. You'd be better using shared memory.
     */
    if constexpr(get_storage_word_count() > 64) {
        storage_type word = sycl::ext::runtime_index_wrapper_log(storage_array_, idx / word_bit_size());
        return sycl::ext::read_bit(word, idx % word_bit_size());
    } else {
        storage_type word = sycl::ext::runtime_index_wrapper(storage_array_, idx / word_bit_size());
        return sycl::ext::read_bit(word, idx % word_bit_size());
    }
}

template<int N, typename storage_type>
constexpr bool register_bit_array<N, storage_type>::operator[](const uint &i) const noexcept {
    return test(i);
}

template<int N, typename storage_type>
constexpr register_bit_array<N, storage_type> &register_bit_array<N, storage_type>::set(const uint &idx) noexcept {
    assume(idx < size());
    sycl::ext::runtime_index_wrapper_transform_ith(
            storage_array_,
            idx / word_bit_size(), /* Word index */
            [&](const storage_type &word) {
                return sycl::ext::set_bit_in_word<true>(word, idx % word_bit_size()); /* Position in word */
            });
    return *this;
}

template<int N, typename storage_type>
constexpr register_bit_array<N, storage_type> &register_bit_array<N, storage_type>::set() noexcept {
    for (auto &word: storage_array_) {
        if constexpr(std::is_same_v<bool, storage_type>) {
            word = true;
        } else {
            word = storage_type{0} - 1;
        }

    }
    return *this;
}

template<int N, typename storage_type>
constexpr register_bit_array<N, storage_type> &register_bit_array<N, storage_type>::reset(const uint &idx) noexcept {
    assume(idx < size());
    sycl::ext::runtime_index_wrapper_transform_ith(
            storage_array_,
            idx / word_bit_size(),
            [&](const storage_type &word) {
                return sycl::ext::set_bit_in_word<false>(word, idx % word_bit_size());
            });
    return *this;
}

template<int N, typename storage_type>
constexpr register_bit_array<N, storage_type> &register_bit_array<N, storage_type>::reset() noexcept {
#pragma unroll
    for (auto &a: storage_array_) {
        a = 0;
    }
    return *this;
}

template<int N, typename storage_type>
constexpr register_bit_array<N, storage_type> &register_bit_array<N, storage_type>::flip(const uint &idx) noexcept {
    assume(idx < size());
    sycl::ext::runtime_index_wrapper_transform_ith(
            storage_array_,
            idx / word_bit_size(),
            [&](const storage_type &word) {
                return sycl::ext::flip_bit(word, idx % word_bit_size());
            });
    return *this;
}

template<int N, typename storage_type>
constexpr register_bit_array<N, storage_type> &register_bit_array<N, storage_type>::write(const uint &idx, bool val) noexcept {
    assume(idx < size());
    if (val) {
        set(idx);
    } else {
        reset(idx);
    }
    return *this;
}

template<int N, typename storage_type>
constexpr uint32_t register_bit_array<N, storage_type>::count() const noexcept {
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

template<int N, typename storage_type>
constexpr bool register_bit_array<N, storage_type>::any() const {
    bool result = false;
    sycl::ext::runtime_index_wrapper_for_all(
            storage_array_,
            [&](const uint &i, const storage_type &word) {
                if constexpr(std::is_same_v<storage_type, bool>) {
                    result = result || word;
                } else {
                    if (i + 1 != get_storage_word_count()) {
                        result = result || (word != 0);
                    } else {
                        auto low_bit_mask = generate_low_bit_mask(size() % word_bit_size());
                        result = result || ((word & low_bit_mask) != 0);
                    }
                }
            });
    return result;
}

template<int N, typename storage_type>
constexpr bool register_bit_array<N, storage_type>::none() const noexcept {
    return !any();
}

template<int N, typename storage_type>
constexpr bool register_bit_array<N, storage_type>::all() const {
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
                        auto low_bit_mask = generate_low_bit_mask(size() % word_bit_size());
                        result = result && ((word & low_bit_mask) == low_bit_mask);
                    }
                }
            });
    return result;
}


template<int N, typename storage_type>
constexpr register_bit_array<N, storage_type> &register_bit_array<N, storage_type>::operator|=(const register_bit_array<N, storage_type> &other) {
    for (int i = 0; i < N; ++i) {
        this->storage_array_[i] |= other.storage_array_[i];
    }
    return *this;
}


template<int N, typename storage_type>
constexpr register_bit_array<N, storage_type> &register_bit_array<N, storage_type>::operator^=(const register_bit_array<N, storage_type> &other) {
    for (int i = 0; i < N; ++i) {
        this->storage_array_[i] ^= other.storage_array_[i];
    }
    return *this;
}


template<int N, typename storage_type>
constexpr register_bit_array<N, storage_type> &register_bit_array<N, storage_type>::operator&=(const register_bit_array<N, storage_type> &other) {
    for (int i = 0; i < N; ++i) {
        this->storage_array_[i] &= other.storage_array_[i];
    }
    return *this;
}


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
        register_bit_array<sieve_size + 1, uint8_t> primes{};

        tmp.set();
        if (!tmp.all()) {
            return primes;
        }

        tmp.reset();
        if (!tmp.none()) {
            return primes;
        }

        for (int i = 0; i < tmp.size(); ++i)
            tmp.set(i);


        for (int p = 2; p * p <= sieve_size; p++) {
            if (tmp.test(p)) {
                for (int i = p * p; i <= sieve_size; i += p)
                    tmp.reset(i);
            }
        }


        for (int p = 2; p < tmp.size(); ++p)
            primes.write(p, tmp[p]);
        return primes;
    }();

    static_assert(!primes_100.all());
    static_assert(primes_100.any());
    static_assert(primes_100.count() == 25);
}
