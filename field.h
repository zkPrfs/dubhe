#pragma once
#include "gsl-lite.hpp"
#include "dubhe_instances.h"
#include <cstdint>
#include <cstdlib>
#include <functional>
extern "C" {
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <wmmintrin.h>
}

namespace field {
class GF2E;
}

field::GF2E dot_product(const std::vector<field::GF2E> &lhs,
                        const std::vector<std::vector<uint8_t>> &rhs, int idx);
field::GF2E dot_product(const std::vector<field::GF2E> &lhs,
                        const std::vector<field::GF2E> &rhs);
field::GF2E dot_product(const std::vector<field::GF2E> &lhs,
                        const gsl::span<field::GF2E> &rhs);

namespace field {
class GF2E {

    public:
  uint64_t data;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
  static std::function<uint64_t(__m128i)> reduce;
#pragma GCC diagnostic pop
  static size_t byte_size;
  static uint64_t modulus;

static uint32_t mul_lut[65536][32];
static bool inited;

public:
  GF2E() : data(0){};
  GF2E(uint64_t data) : data(data) {}
  GF2E(const GF2E &other) = default;
  ~GF2E() = default;
  GF2E &operator=(const GF2E &other) = default;

  void clear() { data = 0; }
  void set_coeff(size_t idx) { data |= (1ULL << idx); }
  GF2E operator+(const GF2E &other) const;
  GF2E &operator+=(const GF2E &other);
  GF2E operator-(const GF2E &other) const;
  GF2E &operator-=(const GF2E &other);
  GF2E operator*(const GF2E &other) const;
  GF2E &operator*=(const GF2E &other);
  bool operator==(const GF2E &other) const;
  bool operator!=(const GF2E &other) const;

  GF2E inverse() const;

  void to_bytes(uint8_t *out) const;
  std::vector<uint8_t> to_bytes() const;
  void from_bytes(uint8_t *in);
  static void init_extension_field(const dubhe_instance_t &instance);

  friend GF2E(::dot_product)(const std::vector<field::GF2E> &lhs,
                             const std::vector<field::GF2E> &rhs);
};

const GF2E &lift_uint8_t(uint8_t value);

std::vector<GF2E> get_first_n_field_elements(size_t n);
std::vector<std::vector<GF2E>>
precompute_lagrange_polynomials(const std::vector<GF2E> &x_values);
std::vector<GF2E> interpolate_with_precomputation(
    const std::vector<std::vector<GF2E>> &precomputed_lagrange_polynomials,
    const std::vector<std::vector<uint8_t>> &y_values, int idx);
std::vector<GF2E> interpolate_with_precomputation(
    const std::vector<std::vector<GF2E>> &precomputed_lagrange_polynomials,
    const std::vector<GF2E> &y_values);
std::vector<GF2E> interpolate_with_precomputation(
    const std::vector<std::vector<GF2E>> &precomputed_lagrange_polynomials,
    const gsl::span<GF2E> &y_values);

void pair_mul(uint16_t a, uint16_t b, uint16_t c, uint16_t d, uint16_t *res0, uint16_t *res1);
void pair_mul(uint16_t a0, uint16_t a1, uint16_t a2,
              uint16_t b0, uint16_t b1, uint16_t b2,
              uint16_t *res0, uint16_t *res1, uint16_t *res2);
void scale_mul(uint16_t a1, uint16_t b1, uint16_t c1, uint16_t x,
        uint16_t a0, uint16_t b0, uint16_t c0,
        uint16_t *res0, uint16_t *res1, uint16_t *res2);
uint16_t mul(uint16_t a, uint16_t b);
uint16_t lift(uint8_t value);

std::vector<GF2E> build_from_roots(const std::vector<GF2E> &roots);
GF2E eval(const std::vector<GF2E> &poly, const GF2E &point);
} // namespace field

std::vector<field::GF2E> operator+(const std::vector<field::GF2E> &lhs,
                                   const std::vector<field::GF2E> &rhs);
std::vector<field::GF2E> &operator+=(std::vector<field::GF2E> &self,
                                     const std::vector<field::GF2E> &rhs);
std::vector<field::GF2E> operator*(const std::vector<field::GF2E> &lhs,
                                   const field::GF2E &rhs);
std::vector<field::GF2E> operator*(const field::GF2E &lhs,
                                   const std::vector<field::GF2E> &rhs);
std::vector<field::GF2E> operator*(const std::vector<field::GF2E> &lhs,
                                   const std::vector<field::GF2E> &rhs);

/*
class F {
    public:
        uint16_t data;
        static bool inited;
        static size_t byte_size;
        static uint32_t mul_lut[(1 << 24) * 2];
        static uint32_t inv_lut[1 << 16];
        static uint8_t MUL_LUT[1 << 16];
        static uint8_t INV_LUT[1 << 8];

    public:
        F() : data(0){};
        F(uint16_t data) : data(data) {}
        F(const F &other) = default;
        ~F() = default;
        F &operator=(const F &other) = default;

        void clear() { data = 0; }
        void set_coeff(size_t idx) { data |= (1 << idx); }
        F operator+(const F &other) const;
        F &operator+=(const F &other);
        F operator-(const F &other) const;
        F &operator-=(const F &other);
        F operator*(const F &other) const;
        F &operator*=(const F &other);
        bool operator==(const F &other) const;
        bool operator!=(const F &other) const;

        F inverse() const;

        void to_bytes(uint8_t *out) const;
        std::vector<uint8_t> to_bytes() const;
        void from_bytes(uint8_t *in);
        static void init_extension_field();

        // friend F(::dot_product)(const std::vector<F> &lhs,
        //         const std::vector<F> &rhs);
};

const F lift_uint8_t(uint8_t value);
std::vector<F> get_first_n_field_elements(size_t n);
std::vector<std::vector<F>>
precompute_lagrange_polynomials(const std::vector<F> &x_values);
std::vector<F> interpolate_with_precomputation(
    const std::vector<std::vector<F>> &precomputed_lagrange_polynomials,
    const std::vector<F> &y_values);
std::vector<F> interpolate_with_precomputation(
    const std::vector<std::vector<F>> &precomputed_lagrange_polynomials,
    const gsl::span<F> &y_values);
std::vector<F> build_from_roots(const std::vector<F> &roots);
F eval(const std::vector<F> &poly, const F &point);
std::vector<F> operator+(const std::vector<F> &lhs,
                                   const std::vector<F> &rhs);
std::vector<F> &operator+=(std::vector<F> &self,
                                     const std::vector<F> &rhs);
std::vector<F> operator*(const std::vector<F> &lhs,
                                   const F &rhs);
std::vector<F> operator*(const F &lhs,
                                   const std::vector<F> &rhs);
std::vector<F> operator*(const std::vector<F> &lhs,
                                   const std::vector<F> &rhs);
F dot_product(const std::vector<F> &lhs,
        const std::vector<F> &rhs);
F dot_product(const std::vector<F> &lhs,
        const gsl::span<F> &rhs);

template <int N>
class vF {
    public:
        __m256i data[N];
        vF();
        vF operator+(const vF &other) const;
        vF &operator+=(const vF &other);
        vF operator-(const vF &other) const;
        vF &operator-=(const vF &other);
        vF operator*(const vF &other) const;
        vF &operator*=(const vF &other);
        // F sum(const vF &other);
};
*/
