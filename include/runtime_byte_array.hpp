/**
 * Array of bytes accessible with runtimes indices and that is stored using larger types to reduce register look-up lacencies/
 */

#pragma once

#include <runtime_index_wrapper.hpp>


template<int N, typename storage_type = uint32_t>
class runtime_byte_array {
public:

    static_assert(std::is_unsigned_v<storage_type> && std::is_integral_v<storage_type>);

    /**
     * Connstructor that takes a list of bytes
     * @param init
     */
    [[nodiscard]] constexpr runtime_byte_array(const std::initializer_list<uint8_t> &init) {
        int idx = 0;
        for (auto b: init) {
            write(idx, b);
            ++idx;
        }
    }


    /**
     * Reads the ith byte
     * @param i index
     * @return the byte
     */
    [[nodiscard]] constexpr uint8_t read(const uint &i) const {
        storage_type word = sycl::ext::runtime_index_wrapper(storage_array_, i / sizeof(storage_type));
        return sycl::ext::get_byte(word, i % sizeof(storage_type));
    }

    /**
     * Reads the ith byte
     * @param i index
     * @return the byte
     */
    [[nodiscard]] constexpr uint8_t operator[](const uint &i) const {
        return read(i);
    }

    /**
     * Write the ith byte
     * @param i index
     * @return the byte written
     */
    constexpr uint8_t write(const uint &i, const uint8_t &write_byte) {
        return sycl::ext::runtime_index_wrapper_store_byte(storage_array_, i / sizeof(storage_type), write_byte, i % sizeof(storage_type));
    }

private:

    static constexpr int get_storage_size() {
        return (N + sizeof(storage_type) - 1) / sizeof(storage_type);
    }

    std::array<storage_type, get_storage_size()> storage_array_{};

};