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
 * @brief Set the mpz integer based on the provided string.
 */
__device__ __host__ inline void mpz_set_str(mpz_t *mpz, const char *str) {
  unsigned num_digits;
  unsigned i;
  int is_zero;

  /* Check if the provided number is negative */
  if (str[0] == '-') {
    mpz->sign = -1;
    str++; // the number starts at the next character
  }
  else {
    mpz->sign = 1;
  }

  num_digits = cuda_strlen(str);
  CHECK_MEM(mpz, num_digits);

  digits_set_zero(mpz->digits);

  is_zero = true;
  for (i = 0; i < num_digits; i++) {
    digit_t d = digit_fromchar(str[num_digits - i - 1]);

    /* keep track of whether or not every digit is zero */
    is_zero = is_zero && (d == 0);

    /* parse the string backwards (little endian order) */
    mpz->digits[i] = d;
  }

  if (is_zero) mpz->sign = 0;
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
__device__ __host__ inline char* mpz_get_str(mpz_t *mpz, char *buf, unsigned bufsize) {
  char *str;
  int print_zeroes = 0; // don't print leading 0s
  int i, str_index = 0;
  int prefix_index = 0;
  int max_size_of_buf = digits_count(mpz->digits) + 1  // for the NULL terminator
                                                  + 1; // for the negative sign

  // for now just assume the user provided buffer is large enough to hold
  // the string representation of the integer...
  (void) bufsize;

  if (NULL == buf) {
    str = (char *) malloc (sizeof(char) * (max_size_of_buf));
  }
  else {
    str = buf;
  }

  if (mpz_is_negative(mpz)) {
    str[0] = '-';
    prefix_index = 1;
  }

  for (i = mpz->capacity - 1; i >= 0; i--) {
    int digit = mpz->digits[i];

    if (digit != 0 || print_zeroes) {
      print_zeroes = 1;
      str[prefix_index + str_index++] = digit_tochar(digit);
    }
  }

  str[prefix_index + str_index] = (char) 0;

  /* the number is zero */
  if (str_index == 0) {
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


__device__ __host__ inline char* mpz_get_binary_str(mpz_t *mpz, char *buf,
                                             unsigned bufsize) {
  char *str;
  int print_zeroes = 0; // don't print leading 0s
  int i, str_index = 0;
  int prefix_index = 0;
  int max_size_of_buf = BINARY_CAPACITY + 1  // for the NULL terminator
                                        + 1; // for the negative sign
  digit_t bits[BINARY_CAPACITY];

  // for now just assume the user provided buffer is large enough to hold
  // the string representation of the integer...
  (void) bufsize;

  if (NULL == buf) {
    str = (char *) malloc (sizeof(char) * (max_size_of_buf));
  }
  else {
    str = buf;
  }

  if (mpz_is_negative(mpz)) {
    str[0] = '-';
    prefix_index = 1;
  }

  digits_to_binary(bits, mpz->digits);

  for (i = BINARY_CAPACITY - 1; i >= 0; i--) {
    int bit = bits[i];

    if (bit != 0 || print_zeroes) {
      print_zeroes = 1;
      str[prefix_index + str_index++] = '0' + bit;
    }
  }

  str[prefix_index + str_index] = (char) 0;

  /* the number is zero */
  if (str_index == 0) {
    str[0] = '0';
    str[1] = (char) 0;
  }

  return str;
}

/**
 * @brief Compute the quotient and remainder of n / d.
 *
 * Credit for the algorithm to compute the quotient goes to Steven S. Skiena.
 * Original source code can be found here:
 *        http://www.cs.sunysb.edu/~skiena/392/programs/bignum.c
 */
__device__ __host__ inline void mpz_div(mpz_t *q, mpz_t *r, mpz_t *n, mpz_t *d) {
  unsigned n_digit_count = digits_count(n->digits);
  mpz_t row;
  mpz_t tmp;
  mpz_t digit;
  int i;
  int nsign = n->sign;
  int dsign = d->sign;

  (void)r;

  mpz_init(&row);
  mpz_init(&tmp);
  mpz_init(&digit);

  mpz_set_i(q, 0);
  mpz_set_i(r, 0);

  if (n->sign < 0) n->sign = 1;
  if (d->sign < 0) d->sign = 1;

  if (mpz_gt(n, d)) {

    for (i = n_digit_count - 1; i >= 0; i--) {
      digits_rshift(row.digits, row.capacity, 1);

      mpz_set_i(&digit, (int) n->digits[i]);
      mpz_add(&tmp, &row, &digit);
      mpz_set(&row, &tmp);

      q->digits[i] = 0;
      while (mpz_gte(&row, d)) {
        q->digits[i]++;

        // row -= d
        mpz_sub(&tmp, &row, d);
        mpz_set(&row, &tmp);
      }
    }

    q->sign = 1;

    mpz_mult(&tmp, q, d);
    mpz_sub(r, n, &tmp);

    q->sign = nsign * dsign;
  }
  else {
    // quotient = 0
    mpz_set_i(q, 0);
    // remainder = numerator
    mpz_set(r, n);
  }

  n->sign = nsign;
  d->sign = dsign;

  CHECK_SIGN(q);
  CHECK_SIGN(r);
  CHECK_SIGN(n);
  CHECK_SIGN(d);
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
  digit_t binary_exp[BINARY_CAPACITY];
  unsigned iteration;
  mpz_t tmp;
  mpz_t ignore;
  mpz_t _base;

  // result = 1
  mpz_set_i(result, 1);

  mpz_init(&tmp);
  mpz_init(&ignore);
  mpz_init(&_base);

  // _base = base % mod
  mpz_set(&tmp, base);
  mpz_div(&ignore, &_base, &tmp, mod);

  digits_to_binary(binary_exp, exp->digits);

  iteration = 0;
  while (!digits_is_zero(binary_exp + iteration, BINARY_CAPACITY - iteration)) {
    // if (binary_exp is odd)
    if (binary_exp[iteration] == 1) {
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

/**
 * @brief Return FLOOR[ log (base DIGIT_BASE) of (op1) ]
 */
__device__ __inline__ void mpz_log_floor(mpz_t *log, mpz_t *op1) {
  mpz_set_lui(log, digits_count(op1->digits) - 1);
}

/**
 * @brief Return CEILING[ log (base DIGIT_BASE) of (op1) ]
 */
__device__ __inline__ void mpz_log_ceil(mpz_t *log, mpz_t *op1) {
  mpz_t one;
  mpz_t minus1;

  mpz_init(&one);
  mpz_init(&minus1);
  mpz_set_i(&one, 1);

  // minus1 = op1 - 1
  mpz_sub(&minus1, op1, &one);
  
  if (digits_is_zero(minus1.digits, minus1.capacity)) {
    mpz_set_lui(log, 0);
  }
  else {
    mpz_set_lui(log, digits_count(minus1.digits));
  }
}

#endif /* __418_MPZ_H__ */
