

int constexpr gcd(int a, int b) {
    while (a != b) {
        if (a > b) {
            a -= b;
        } else {
            b -= a;
        }
    }
    return a;
}


template<typename T, int bank_byte_size = 4, int bank_count = 32>
void constexpr assert_local_alignement() {
    static_assert(sizeof(T) <= bank_byte_size || (sizeof(T) > bank_byte_size && sizeof(T) % bank_byte_size == 0), "Must be 4 bytes aligned");
    static_assert(gcd(sizeof(T) / bank_byte_size, bank_count) == 1, "Must pad in order to have GCD(sizeof(T)/bank_size), bank_count)==1");
}