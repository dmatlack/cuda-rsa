/**
 * @file mpz.c
 *
 * @brief Multiple Precision arithmetic code.
 *
 * @author David Matlack (dmatlack)
 */
#ifndef __418_MPZ_H__
#define __418_MPZ_H__

#define true 1
#define false 0

#ifndef __CUDACC__ /* when compiling with g++ ... */

#define __device__
#define __host__

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

inline unsigned max(unsigned a, unsigned b) { return (a > b) ? a : b; }
inline unsigned min(unsigned a, unsigned b) { return (a < b) ? a : b; }
inline int abs(int a) { return (a < 0) ? -a : a; }

#else /* when compiling with nvcc ... */

#endif /* __CUDACC__ */

#include "cuda_string.h"
#include "digit.h"

#define DEFAULT_INITIAL_MAX_DIGITS 64

/** @breif struct used to represent multiple precision integers (Z) */
typedef struct {
  digit_t     *digits;       // an array of digits, in little endian order
  unsigned     max_digits;   // the capacity of the digits array
  int          sign;         // +1, 0, or -1
} mpz_t;

/** @brief Called if the program runs out of memory */
__device__ __host__ void mpz_memory_error(const char *function) {
#ifndef __CUDACC__
  printf("Unable to allocate memory in %s!\n", function);
  exit(42);
#else
  //TODO
#endif
}

/**
 * @brief Count the number of digits (actually being used) in the mpz
 * struct.
 *
 * For example, 0-9       = 1 digit, 
 *              1000-9999 = 4 digits, 
 *              etc.
 *
 */
__device__ __host__ unsigned mpz_count_digits(mpz_t *mpz) {
  int is_leading_zero = 1;
  int count = 0;
  int i;

  for (i = mpz->max_digits - 1; i >= 0; i--) {
    digit_t d = mpz->digits[i];

    if (0 == d && is_leading_zero) continue;

    is_leading_zero = 0;
    count++;
  }

  /* special case where all digits are 0 */
  if (count == 0) return 1;

  return count;
}

inline __device__ __host__ int mpz_is_negative(mpz_t *mpz) {
  if (digits_is_zero(mpz->digits, mpz->max_digits)) return false;
  return mpz->sign == -1;
}

inline __device__ __host__ void mpz_negate(mpz_t *mpz) {
  mpz->sign *= -1;
}


/**
 * @brief Make the size of the mpz_t struct the first power of 2 that is
 * greater than or equal to new_max_digits.
 *
 * @warning If new_max_digits is less than the size of the mpz struct than
 * some information may be lost!
 *
 * We currently only grow the mpz structs as we need more room for digits. 
 * Maybe in the future we will want to shrink the digit array? Probably not
 * though.
 */
__device__ __host__ void mpz_ensure_mem(mpz_t *mpz, unsigned new_max_digits) {
  digit_t *new_digits;
  unsigned i;

  if (mpz->max_digits < new_max_digits) {
    /* Find the next power of 2 digits that gives us at least new_max_digits */
    unsigned find_new_max = 2;
    while (find_new_max < new_max_digits) find_new_max *= 2;
    new_max_digits = find_new_max;

    new_digits = (digit_t *) malloc (new_max_digits * sizeof(digit_t));
    if (NULL == new_digits) mpz_memory_error("mpz_alloc");

    /* Copy the old data over */
    for (i = 0; i < new_max_digits; i++) {
      new_digits[i] = (i < mpz->max_digits) ? mpz->digits[i] : 0;
    }

    /* Free the old digits array */
    if (mpz->max_digits > 0) free(mpz->digits);

    mpz->digits = new_digits;
    mpz->max_digits = new_max_digits;
  }

}

/**
 * @brief Set the mpz_t struct to zero.
 */
__device__ __host__ void mpz_clear(mpz_t *mpz) {
  unsigned i;

  for (i = 0; i < mpz->max_digits; i++) {
    mpz->digits[i] = 0;
  }
}

/**
 * @brief If the number is zero, the sign should be zero.
 */
static inline __device__ __host__ void _assert_sign(mpz_t *mpz) {
#ifndef __CUDACC__
  if (digits_is_zero(mpz->digits, mpz->max_digits)) {
    assert(0 == mpz->sign);
  }
  else {
    assert(1 == mpz->sign || -1 == mpz->sign);
  }
#endif
}

__device__ __host__ void mpz_init(mpz_t *mpz) {
  mpz->digits = (digit_t *) malloc 
                (sizeof(digit_t) * DEFAULT_INITIAL_MAX_DIGITS);

  if (mpz->digits == NULL) mpz_memory_error("mpz_init");

  mpz->max_digits = DEFAULT_INITIAL_MAX_DIGITS;

  mpz_clear(mpz);

  mpz->sign = 0;
}

__device__ __host__ void mpz_set(mpz_t *to, mpz_t *from) {
  unsigned from_digits = mpz_count_digits(from);
  unsigned i;

  mpz_ensure_mem(to, from_digits);
  mpz_clear(to);

  for (i = 0; i < to->max_digits; i++) {
    digit_t d = (i < from->max_digits) ? from->digits[i] : 0;
    to->digits[i] = d;
  }

  to->sign = from->sign;
}

/**
 * @brief Set the mpz integer to the provided integer.
 *
 * @warning Assumes the mpz struct has been initialized.
 */
__device__ __host__ void mpz_set_i(mpz_t *mpz, int z) {
  unsigned i;

  mpz_ensure_mem(mpz, 10);
  mpz_clear(mpz);

  mpz->sign = (z == 0) ? 0 : ((z > 0) ? 1: -1);

  z = abs(z);

  i = 0;
  while (z > 0 && i < mpz->max_digits) {
    mpz->digits[i++] = z % 10;
    z /= 10;
  }
}

/**
 * @brief Set the mpz integer based on the provided string.
 *
 * @warning Assumes the mpz struct has been initialized.
 */
__device__ __host__ void mpz_set_str(mpz_t *mpz, const char *str) {
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

  mpz_ensure_mem(mpz, num_digits);
  mpz_clear(mpz);

  is_zero = 1;
  for (i = 0; i < num_digits; i++) {
    digit_t d = digit_fromchar(str[num_digits - i - 1]);

    is_zero = is_zero && (d == 0);

    /* parse the string backwards (little endian order) */
    mpz->digits[i] = d;
  }

  if (is_zero) mpz->sign = 0;
}

/**
 * @breif Destroy the mpz_t struct.
 */
__device__ __host__ void mpz_destroy(mpz_t *mpz) {
  free(mpz->digits);
  mpz->max_digits = 0;
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
__device__ __host__ void mpz_add(mpz_t *dst, mpz_t *op1, mpz_t *op2) {
  unsigned op1_digit_count = mpz_count_digits(op1);
  unsigned op2_digit_count = mpz_count_digits(op2);

  /* In addition, if the operand with the most digits has D digits, then
   * the result of the addition will have at most D + 1 digits. */
  unsigned max_digits = max(op1_digit_count, op2_digit_count) + 1;

  /* Make sure all of the mpz structs have enough memory to hold all of
   * the digits. We will be doing 10's complement so everyone needs to 
   * have enough digits. */
  mpz_ensure_mem(dst,  max_digits);
  mpz_ensure_mem(op1, max_digits);
  mpz_ensure_mem(op2, max_digits);

  mpz_clear(dst);

  /* If both are negative, treate them as positive and negate the result */
  if (mpz_is_negative(op1) && mpz_is_negative(op2)) {
    digits_add(dst->digits, dst->max_digits, 
               op1->digits, op1->max_digits,
               op2->digits, op2->max_digits);
    dst->sign = -1;
  }
  /* one or neither are negative */
  else {
    digit_t carry_out;

    /* Perform 10's complement on negative numbers before adding */
    if (mpz_is_negative(op1)) digits_complement(op1->digits, op1->max_digits);
    if (mpz_is_negative(op2)) digits_complement(op2->digits, op2->max_digits);

    carry_out = digits_add(dst->digits, dst->max_digits, 
                           op1->digits, op1->max_digits,
                           op2->digits, op2->max_digits);
    
    /* If there is no carryout, the result is negative */
    if (carry_out == 0 && (mpz_is_negative(op1) || mpz_is_negative(op2))) {
      digits_complement(dst->digits, dst->max_digits);
      dst->sign = -1;
    }
    /* Otherwise, the result is non-negative */
    else {
      dst->sign = (digits_is_zero(dst->digits, dst->max_digits)) ? 0 : 1;
    }

    /* Undo the 10s complement after adding */
    if (mpz_is_negative(op1)) digits_complement(op1->digits, op1->max_digits);
    if (mpz_is_negative(op2)) digits_complement(op2->digits, op2->max_digits);
  }

  // FIXME remove eventually
  _assert_sign(op1);
  _assert_sign(op2);
  _assert_sign(dst);
}

/**
 * @brief Perform dst := op1 - op2.
 *
 * @warning Assumes that all mpz_t parameters have been initialized.
 * @warning Assumes dst != op1 != op2
 */
__device__ __host__ void mpz_sub(mpz_t *dst, mpz_t *op1, mpz_t *op2) {
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
__device__ __host__ void mpz_mult(mpz_t *dst, mpz_t *op1, mpz_t *op2) {
  unsigned op1_digit_count = mpz_count_digits(op1);
  unsigned op2_digit_count = mpz_count_digits(op2);
  unsigned max_digits = max(op1_digit_count, op2_digit_count);

  /* In multiplication, if the operand with the most digits has D digits, 
   * then the result of the addition will have at most 2D digits. */
  mpz_ensure_mem(dst, 2*max_digits);
  mpz_ensure_mem(op1,   max_digits);
  mpz_ensure_mem(op2,   max_digits);

  /* We pass in max_digits as the number of digits rather that the actual
   * number of digits in each mpz_t struct. This is done because the 
   * multiplication code has some assumptions and optimizations (e.g.
   * op1 and op2 to have the same number of digits) */
  digits_mult(dst->digits, op1->digits, op2->digits, max_digits);

  /* Compute the sign of the product */
  dst->sign = op1->sign * op2->sign;

  // TODO remove eventually
  _assert_sign(op1);
  _assert_sign(op2);
  _assert_sign(dst);
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
__device__ __host__ int mpz_compare(mpz_t *a, mpz_t *b) {
  int cmp;
  int negative;

  if (a->sign < b->sign) return MPZ_LESS;
  if (a->sign > b->sign) return MPZ_GREATER;
 
  /* At this point we know they have the same sign */
  cmp = digits_compare(a->digits, a->max_digits, b->digits, b->max_digits);
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
__device__ __host__ int mpz_equal(mpz_t *a, mpz_t *b) {
  return (mpz_compare(a, b) == 0);
}
/** @brief Return true if a < b */
__device__ __host__ int mpz_lt(mpz_t *a, mpz_t *b) {
  return (mpz_compare(a, b) < 0);
}
/** @brief Return true if a <= b */
__device__ __host__ int mpz_lte(mpz_t *a, mpz_t *b) {
  return (mpz_compare(a, b) <= 0);
}
/** @brief Return true if a > b */
__device__ __host__ int mpz_gt(mpz_t *a, mpz_t *b) {
  return (mpz_compare(a, b) > 0);
}
/** @brief Return true if a >= b */
__device__ __host__ int mpz_gte(mpz_t *a, mpz_t *b) {
  return (mpz_compare(a, b) >= 0);
}

/**
 * @breif Return the string representation of the integer represented by the
 * mpz_t struct.
 *
 * @warning If buf is NULL, the string is dynamically allocated and must 
 * therefore be freed by the user.
 */
__device__ __host__ char* mpz_get_str(mpz_t *mpz, char *buf, unsigned bufsize) {
  char *str;
  int print_zeroes = 0; // don't print leading 0s
  int i, str_index = 0;
  int prefix_index = 0;
  int max_size_of_buf = mpz_count_digits(mpz) + 1  // for the NULL terminator
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

  for (i = mpz->max_digits - 1; i >= 0; i--) {
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

#endif /* __418_MPZ_H__ */
