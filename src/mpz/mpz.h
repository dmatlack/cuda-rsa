/**
 * @file mpz.c
 *
 * @brief Multiple Precision arithmetic code.
 *
 * @author David Matlack (dmatlack)
 */
#ifndef __418_MPZ_H__
#define __418_MPZ_H__

#include "compile.h"
#include "cuda_string.h"
#include "digit.h"


/** @breif struct used to represent multiple precision integers (Z). */
typedef struct {
  digit_t  digits[DIGITS_CAPACITY];
  unsigned capacity;
  char     sign;
} mpz_t;

__device__ __host__ inline char* mpz_get_str(mpz_t *mpz, char *str, int bufsize);

/**
 * @brief Check that the mpz_t struct has enough memory to store __capacity
 * digits.
 */
#ifndef __CUDACC__
#define CHECK_MEM(__mpz, __capacity) \
  do {                                                                  \
    if ((__mpz)->capacity < (__capacity)) {                             \
      printf("MPZ memory error at %s:%d.\n", __func__, __LINE__);       \
      printf("\tmpz capacity: %u, requested capacity %u\n",             \
             (__mpz)->capacity, (__capacity));                          \
      exit(4);                                                          \
    }                                                                   \
  } while (0)
#else
#define CHECK_MEM(__mpz, __capacity)
#endif

/**
 * @brief Do some sanity checking on the mpz_t sign field.
 */
#ifndef __CUDACC__
#define CHECK_SIGN(__mpz) \
  do {                                                                \
    if (digits_is_zero((__mpz)->digits, (__mpz)->capacity)) {         \
      if (0 != (__mpz)->sign) {                                       \
        printf("Sign should be 0 in %s:%d but is %d instead.\n",      \
             __func__, __LINE__, (__mpz)->sign);                      \
        printf("\tmpz = ");                                           \
        digits_print((__mpz)->digits, (__mpz)->capacity);             \
        printf("\n");                                                 \
      }                                                               \
    }                                                                 \
    else if (1 != (__mpz)->sign && -1 != (__mpz)->sign) {             \
      printf("Sign should be 1 or -1 in %s:%d but is %d instead.\n",  \
             __func__, __LINE__, (__mpz)->sign);                      \
    }                                                                 \
  } while (0)
#else
#define CHECK_SIGN(__mpz)
#endif

/**
 * @brief Do some sanity checking on the mpz_t sign field.
 */
#ifndef __CUDACC__
#define CHECK_STRS(s1, s2)                                            \
  do {                                                                \
    if (strcmp(s1, s2)) {                                             \
      printf("Input string %s became %s!\n", s1, s2);                 \
    }                                                                 \
  } while (0)
#else
#define CHECK_STRS(s1, s2)
#endif


__device__ __host__ inline int mpz_is_negative(mpz_t *mpz) {
  if (digits_is_zero(mpz->digits, mpz->capacity)) return false;
  return (mpz->sign == -1);
}

__device__ __host__ inline void mpz_negate(mpz_t *mpz) {
  mpz->sign *= -1;
}

/**
 * @brief Initialize an mpz struct to 0.
 */
__device__ __host__ inline void mpz_init(mpz_t *mpz) {
  mpz->capacity = DIGITS_CAPACITY;
  digits_set_zero(mpz->digits);
  mpz->sign = 0;
}

/**
 * @brief Assign an mpz_t struct to the value of another mpz_t struct.
 */
__device__ __host__ inline void mpz_set(mpz_t *to, mpz_t *from) {
  unsigned i;

  for (i = 0; i < to->capacity; i++) {
    digit_t d = (i < from->capacity) ? from->digits[i] : 0;
    to->digits[i] = d;
  }

  to->sign = from->sign;
}

/**
 * @brief Set the mpz integer to the provided integer.
 */
__device__ __host__ inline void mpz_set_i(mpz_t *mpz, int z) {
  mpz->sign = (z == 0) ? 0 : ((z > 0) ? 1: -1);
  digits_set_lui(mpz->digits, abs(z));
}

/**
 * @brief Set the mpz integer to the provided integer.
 */
__device__ __host__ inline void mpz_set_lui(mpz_t *mpz, unsigned long z) {
  mpz->sign = (z == 0) ? 0 : ((z > 0) ? 1: -1);
  digits_set_lui(mpz->digits, z);
}

/**
 * @brief Set the mpz integer based on the provided (hex) string.
 */
__device__ __host__ inline void mpz_set_str(mpz_t *mpz, const char *user_str) {
  unsigned num_digits;
  unsigned i;
  int is_zero;

  for (i = 0; i < mpz->capacity; i++) mpz->digits[i] = 0;

  const int bufsize = 1024;
  char buf[bufsize];
  memcpy(buf, user_str, bufsize);
  buf[bufsize - 1] = (char) 0;
  char *str = &buf[0];

  /* Check if the provided number is negative */
  if (str[0] == '-') {
    mpz->sign = -1;
    str ++; // the number starts at the next character
  }
  else {
    mpz->sign = 1;
  }

  int len = cuda_strlen(str);
  int char_per_digit = LOG2_DIGIT_BASE / 4;
  num_digits = (len + char_per_digit - 1) / char_per_digit;
  CHECK_MEM(mpz, num_digits);

  digits_set_zero(mpz->digits);

  is_zero = true;
  for (i = 0; i < num_digits; i ++) {
    str[len - i * char_per_digit] = (char) 0;
    char *start = str + (int) max(len - (i + 1) * char_per_digit, 0);
    digit_t d = strtol(start, NULL, 16);

    /* keep track of whether or not every digit is zero */
    is_zero = is_zero && (d == 0);

    /* parse the string backwards (little endian order) */
    mpz->digits[i] = d;
  }

  if (is_zero) mpz->sign = 0;

#if 0
  mpz_get_str(mpz, buf, bufsize);
  CHECK_STRS(user_str, buf);
#endif
}

__device__ __host__ inline void mpz_get_binary_str(mpz_t *mpz, char *str, unsigned s) {
  (void) mpz;
  (void) str;
  (void) s;
}

/**
 * @breif Destroy the mpz_t struct.
 */
__device__ __host__ inline void mpz_destroy(mpz_t *mpz) {
  mpz->sign = 0;
}

/**
 * @brief Add two multiple precision integers.
 *
 *      dst := op1 + op2
 *
 * @warning It is assumed that all mpz_t parameters have been initialized.
 * @warning Assumes dst != op1 != op2
 */
__device__ __host__ inline void mpz_add(mpz_t *dst, mpz_t *op1, mpz_t *op2) {
  unsigned op1_digit_count = digits_count(op1->digits);
  unsigned op2_digit_count = digits_count(op2->digits);

  /* In addition, if the operand with the most digits has D digits, then
   * the result of the addition will have at most D + 1 digits. */
  unsigned capacity = max(op1_digit_count, op2_digit_count) + 1;

  /* Make sure all of the mpz structs have enough memory to hold all of
   * the digits. We will be doing 10's complement so everyone needs to
   * have enough digits. */
  CHECK_MEM(dst, capacity);
  CHECK_MEM(op1, capacity);
  CHECK_MEM(op2, capacity);

  digits_set_zero(dst->digits);

  /* If both are negative, treate them as positive and negate the result */
  if (mpz_is_negative(op1) && mpz_is_negative(op2)) {
    digits_add(dst->digits, dst->capacity,
               op1->digits, op1->capacity,
               op2->digits, op2->capacity);
    dst->sign = -1;
  }
  /* one or neither are negative */
  else {
    digit_t carry_out;

    /* Perform 10's complement on negative numbers before adding */
    if (mpz_is_negative(op1)) digits_complement(op1->digits, op1->capacity);
    if (mpz_is_negative(op2)) digits_complement(op2->digits, op2->capacity);

    carry_out = digits_add(dst->digits, dst->capacity,
                           op1->digits, op1->capacity,
                           op2->digits, op2->capacity);

    /* If there is no carryout, the result is negative */
    if (carry_out == 0 && (mpz_is_negative(op1) || mpz_is_negative(op2))) {
      digits_complement(dst->digits, dst->capacity);
      dst->sign = -1;
    }
    /* Otherwise, the result is non-negative */
    else {
      dst->sign = (digits_is_zero(dst->digits, dst->capacity)) ? 0 : 1;
    }

    /* Undo the 10s complement after adding */
    if (mpz_is_negative(op1)) digits_complement(op1->digits, op1->capacity);
    if (mpz_is_negative(op2)) digits_complement(op2->digits, op2->capacity);
  }

  CHECK_SIGN(op1);
  CHECK_SIGN(op2);
  CHECK_SIGN(dst);
}

/**
 * @brief Perform dst := op1 - op2.
 *
 * @warning Assumes that all mpz_t parameters have been initialized.
 * @warning Assumes dst != op1 != op2
 */
__device__ __host__ inline void mpz_sub(mpz_t *dst, mpz_t *op1, mpz_t *op2) {
  mpz_negate(op2);
  mpz_add(dst, op1, op2);
  mpz_negate(op2);
}

/**
 * @brief Perform dst := op1 * op2.
 *
 * @warning Assumes that all mpz_t parameters have been initialized.
 * @warning Assumes dst != op1 != op2
 */
__device__ __host__ inline void mpz_mult(mpz_t *dst, mpz_t *op1, mpz_t *op2) {
  unsigned op1_digit_count = digits_count(op1->digits);
  unsigned op2_digit_count = digits_count(op2->digits);
  unsigned capacity = max(op1_digit_count, op2_digit_count);

  digits_set_zero(dst->digits);

  /* In multiplication, if the operand with the most digits has D digits,
   * then the result of the addition will have at most 2D digits. */
  CHECK_MEM(dst, 2*capacity);
  CHECK_MEM(op1,   capacity);
  CHECK_MEM(op2,   capacity);

  /* We pass in capacity as the number of digits rather that the actual
   * number of digits in each mpz_t struct. This is done because the
   * multiplication code has some assumptions and optimizations (e.g.
   * op1 and op2 to have the same number of digits) */
  digits_mult(dst->digits, op1->digits, op2->digits, capacity);

  /* Compute the sign of the product */
  dst->sign = op1->sign * op2->sign;

  CHECK_SIGN(op1);
  CHECK_SIGN(op2);
  CHECK_SIGN(dst);
}

/**
 * @return
 *      < 0  if a < b
 *      = 0  if a = b
 *      > 0  if a > b
 *
 * @warning This function does not give any indication about the distance
 * between a and b, just the relative distance (<, >, =).
 */
#define MPZ_LESS    -1
#define MPZ_GREATER  1
#define MPZ_EQUAL    0
__device__ __host__ inline int mpz_compare(mpz_t *a, mpz_t *b) {
  int cmp;
  int negative;

  if (a->sign < b->sign) return MPZ_LESS;
  if (a->sign > b->sign) return MPZ_GREATER;

  /* At this point we know they have the same sign */
  cmp = digits_compare(a->digits, a->capacity, b->digits, b->capacity);
  negative = mpz_is_negative(a);

  if (cmp == 0) return MPZ_EQUAL;

  if (negative) {
    return (cmp > 0) ? MPZ_LESS : MPZ_GREATER;
  }
  else {
    return (cmp < 0) ? MPZ_LESS : MPZ_GREATER;
  }
}

/** @brief Return true if a == b */
__device__ __host__ inline int mpz_equal(mpz_t *a, mpz_t *b) {
  return (mpz_compare(a, b) == 0);
}
/** @brief Return true if a < b */
__device__ __host__ inline int mpz_lt(mpz_t *a, mpz_t *b) {
  return (mpz_compare(a, b) < 0);
}
/** @brief Return true if a <= b */
__device__ __host__ inline int mpz_lte(mpz_t *a, mpz_t *b) {
  return (mpz_compare(a, b) <= 0);
}
/** @brief Return true if a > b */
__device__ __host__ inline int mpz_gt(mpz_t *a, mpz_t *b) {
  return (mpz_compare(a, b) > 0);
}
/** @brief Return true if a >= b */
__device__ __host__ inline int mpz_gte(mpz_t *a, mpz_t *b) {
  return (mpz_compare(a, b) >= 0);
}

/**
 * @breif Return the string representation of the integer represented by the
 * mpz_t struct.
 *
 * @warning If buf is NULL, the string is dynamically allocated and must
 * therefore be freed by the user.
 */
__device__ __host__ inline char* mpz_get_str(mpz_t *mpz, char *str, int bufsize) {
  int print_zeroes = 0; // don't print leading 0s
  int i;
  int str_index = 0;

  if (mpz_is_negative(mpz)) {
    str[0] = '-';
    str_index = 1;
  }

  for (i = mpz->capacity - 1; i >= 0; i--) {
    unsigned digit = mpz->digits[i];

    if (digit != 0 || print_zeroes) {
      if (bufsize < str_index + 8) {
        return NULL;
      }
      if (!print_zeroes) {
        str_index += sprintf(str + str_index, "%x", digit);
      }
      else {
        str_index += sprintf(str + str_index, "%08x", digit);
      }
      print_zeroes = 1;
    }
  }

  str[str_index] = (char) 0;

  /* the number is zero */
  if (print_zeroes == 0) {
    str[0] = '0';
    str[1] = (char) 0;
  }

  return str;
}

__device__ __host__ inline void mpz_print(mpz_t *mpz) {
#ifndef __CUDACC__
  char str[1024];

  mpz_get_str(mpz, str, 1024);
  printf("%s", str);
#endif
}

__device__ __host__ inline void mpz_set_bit(mpz_t *mpz, unsigned bit_offset, 
                                            unsigned bit) {
  digits_set_bit(mpz->digits, bit_offset, bit);
  if (0 == mpz->sign && 0 != bit) {
    mpz->sign = 1;
  }
  else if (mpz->sign != 0 && bit == 0) {
    if (digits_is_zero(mpz->digits, mpz->capacity)) mpz->sign = 0;
  }
}

__device__ __host__ inline void mpz_bit_lshift(mpz_t *mpz) {
  bits_lshift(mpz->digits, mpz->capacity);

  if (mpz->sign != 0 && digits_is_zero(mpz->digits, mpz->capacity)) {
    mpz->sign = 0;
  }
}

/**
 * @brief Compute the quotient and remainder of n / d.
 *
 *
 */
__device__ __host__ inline void mpz_div(mpz_t *q, mpz_t *r, mpz_t *N, mpz_t *D) {
  unsigned n_digit_count = digits_count(N->digits);
  unsigned num_bits;
  mpz_t n;
  mpz_t d;
  mpz_t tmp;
  int i;
  int nsign = N->sign;
  int dsign = D->sign;

  (void)r;

  num_bits = n_digit_count * LOG2_DIGIT_BASE;

  mpz_init(&n);
  mpz_init(&d);
  mpz_init(&tmp);

  mpz_set(&n, N);
  mpz_set(&d, D);
  mpz_set_i(q, 0);
  mpz_set_i(r, 0);

  if (n.sign < 0) n.sign = 1;
  if (d.sign < 0) d.sign = 1;

  if (mpz_gt(&n, &d)) {

    for (i = num_bits - 1; i >= 0; i--) {
      unsigned n_i;

      // r = r << 1
      mpz_bit_lshift(r);

      // r(0) = n(i)
      n_i = digits_bit_at(n.digits, i);
      mpz_set_bit(r, 0, n_i);

      // if (r >= d)
      if (mpz_gte(r, &d)) {
        // r = r - d
        mpz_sub(&tmp, r, &d);
        mpz_set(r, &tmp);

        // q(i) = 1
        //printf("Setting bit %d of q to 1\n", i);
        //printf("\tBefore: "); mpz_print(q); printf("\n");
        mpz_set_bit(q, i, 1); 
        //printf("\tAfter: "); mpz_print(q); printf("\n");
      }
    }

    r->sign = 1;
    q->sign = nsign * dsign;
  }
  else {
    // quotient = 0
    mpz_set_i(q, 0);
    // remainder = numerator
    mpz_set(r, &n);
  }

  n.sign = nsign;
  d.sign = dsign;

  if (digits_is_zero(q->digits, q->capacity)) q->sign = 0;
  if (digits_is_zero(r->digits, r->capacity)) r->sign = 0;

  CHECK_SIGN(q);
  CHECK_SIGN(r);
  CHECK_SIGN(N);
  CHECK_SIGN(D);
}

/**
 * @brief Compute the GCD of op1 and op2.
 *
 * Euclidean Algorithm:
 *
 *    while (b != 0) {
 *      t := b
 *      b := a % b
 *      a := t
 *    }
 *    gcd = a
 */
__device__ __inline__ void mpz_gcd(mpz_t *gcd, mpz_t *op1, mpz_t *op2) {
  mpz_t a;
  mpz_t b;
  mpz_t mod;
  mpz_t quo;
  int compare = mpz_compare(op1, op2);

  mpz_init(&a);
  mpz_init(&b);
  mpz_init(&mod);
  mpz_init(&quo);

  mpz_set(&a, (compare > 0) ? op1 : op2);
  mpz_set(&b, (compare > 0) ? op2 : op1);

  while (!digits_is_zero(b.digits, b.capacity)) {
    mpz_div(&quo, &mod, &a, &b);
    mpz_set(&a, &b);
    mpz_set(&b, &mod);
  }

  mpz_set(gcd, &a);
}

/**
 * Using exponentiation by squaring algorithm:
 *
 *  function modular_pow(base, exponent, modulus)
 *    result := 1
 *    while exponent > 0
 *      if (exponent mod 2 == 1):
 *         result := (result * base) mod modulus
 *      exponent := exponent >> 1
 *      base = (base * base) mod modulus
 *    return result
 */
__device__ __inline__ void mpz_powmod(mpz_t *result, mpz_t *base,
                                      mpz_t *exp, mpz_t *mod) {
  unsigned iteration;
  mpz_t e;
  mpz_t tmp;
  mpz_t ignore;
  mpz_t _base;


  // result = 1
  mpz_set_i(result, 1);

  mpz_init(&e);
  mpz_init(&tmp);
  mpz_init(&ignore);
  mpz_init(&_base);

  // e = exp
  mpz_set(&e, exp);

  // _base = base % mod
  mpz_set(&tmp, base);
  mpz_div(&ignore, &_base, &tmp, mod);

  iteration = 0;
  while (!bits_is_zero(e.digits, e.capacity, iteration)) {
    // if (binary_exp is odd)
    if (digits_bit_at(e.digits, iteration) == 1) {
      // result = (result * base) % mod
      mpz_mult(&tmp, result, &_base);
      mpz_div(&ignore, result, &tmp, mod);
    }

    // binary_exp = binary_exp >> 1
    iteration++;

    // base = (base * base) % mod
    mpz_set(&ignore, &_base);
    mpz_mult(&tmp, &_base, &ignore);
    mpz_div(&ignore, &_base, &tmp, mod);
  }
}

#endif /* __418_MPZ_H__ */
