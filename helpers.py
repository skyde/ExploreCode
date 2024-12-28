
# ------------------ Debug Mode Hard-Coded Responses ------------------ #

# A simple valid C++ file
DEBUG_VALID_CODE = r"""
#include <iostream>
#include <cassert>
#include <array>
#include <algorithm>
#include <numeric>

template <typename T, std::size_t N>
class SIMDStructOfArrays {
public:
    SIMDStructOfArrays() = default;

    void set(std::size_t index, const std::array<T, N>& values) {
        assert(index < N);
        data[index] = values;
    }

    std::array<T, N> get(std::size_t index) const {
        assert(index < N);
        return data[index];
    }

    std::array<T, N> sum() const {
        std::array<T, N> result{};
        for (const auto& arr : data) {
            for (std::size_t i = 0; i < N; ++i) {
                result[i] += arr[i];
            }
        }
        return result;
    }

    std::array<T, N> multiply(const std::array<T, N>& factors) const {
        std::array<T, N> result{};
        for (const auto& arr : data) {
            for (std::size_t i = 0; i < N; ++i) {
                result[i] += arr[i] * factors[i];
            }
        }
        return result;
    }

    void clear() {
        for (auto& arr : data) {
            arr.fill(T{});
        }
    }

private:
    std::array<std::array<T, N>, N> data{};
};

void run_tests() {
    SIMDStructOfArrays<int, 3> simd;

    // Test setting and getting values
    simd.set(0, {1, 2, 3});
    simd.set(1, {4, 5, 6});
    simd.set(2, {7, 8, 9});

    assert((simd.get(0) == std::array<int, 3>{1, 2, 3}));
    assert((simd.get(1) == std::array<int, 3>{4, 5, 6}));
    assert((simd.get(2) == std::array<int, 3>{7, 8, 9}));

    // Test sum
    assert((simd.sum() == std::array<int, 3>{12, 15, 18}));

    // Test multiplication
    assert((simd.multiply({1, 1, 1}) == std::array<int, 3>{12, 15, 18}));
    assert((simd.multiply({2, 2, 2}) == std::array<int, 3>{24, 30, 36}));

    // Test clear
    simd.clear();
    assert((simd.get(0) == std::array<int, 3>{0, 0, 0}));
    assert((simd.get(1) == std::array<int, 3>{0, 0, 0}));
    assert((simd.get(2) == std::array<int, 3>{0, 0, 0}));

    // Test out of bounds
    bool caught_exception = false;
    try {
        simd.get(3); // Out of bounds
    } catch (const std::out_of_range&) {
        caught_exception = true;
    }
    assert(caught_exception);

    std::cout << "All tests passed!" << std::endl;
}

int main() {
    run_tests();
    return 0;
}
"""
