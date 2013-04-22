/**
 * @file mpz.c
 *
 * @brief Multiple Precision arithmetic code.
 *
 * @author David Matlack (dmatlack)
 */
#ifndef __418_MPZ_H__
#define __418_MPZ_H__

#ifndef __CUDACC__ /* when compiling with g++ ... */

#define __device__
#define __host__

#include <stdlib.h>
#include <stdio.h>

inline unsigned max(unsigned a, unsigned b) { return (a > b) ? a : b; }
inline unsigned min(unsigned a, unsigned b) { return (a < b) ? a : b; }
inline int abs(int a) { return (a < 0) ? -a : a; }

#else /* when compiling with nvcc ... */

#endif

#include "cuda_string.h"
#include "digit.h"

#define DEFAULT_INITIAL_MAX_DIGITS 64

/** @breif struct used to represent multiple precision integers (Z) */
typedef struct {
  digit_t     *digits;       // an array of digits, in little endian order
  unsigned     max_digits;   // the capacity of the digits array
  int          sign;         // +1 or -1
} mpz_t;

/** @brief Called if the program runs out of memory */
__device__ __host__ void mpz_memory_error(const char *function) {
#ifndef __CUDACC__
  printf("Unable to allocate memory in %s!!!!\n", function);
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
  return mpz->sign == -1;
}

inline __device__ __host__ void mpz_negate(mpz_t *mpz) {
  mpz->sign *= -1;
}


/**
 * @brief Check that the given mpz_t struct has room for num_digits. If not,
 * allocate memory for enough digits and copy the old contents to the new
 * array of digits.
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
    unsigned find_new_max = DEFAULT_INITIAL_MAX_DIGITS;
    while (find_new_max < new_max_digits) find_new_max *= 2;
    new_max_digits = find_new_max;

    new_digits = (digit_t *) malloc (new_max_digits * sizeof(digit_t));

    if (NULL == new_digits) mpz_memory_error("mpz_alloc");

    /* Copy the old data over */
    for (i = 0; i < new_max_digits; i++) {
      if (i < mpz->max_digits) new_digits[i] = mpz->digits[i];
      else mpz->digits[i] = 0;
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

__device__ __host__ void mpz_init(mpz_t *mpz) {
  mpz->digits = (digit_t *) malloc 
                (sizeof(digit_t) * DEFAULT_INITIAL_MAX_DIGITS);

  if (mpz->digits == NULL) mpz_memory_error("mpz_init");

  mpz->max_digits = DEFAULT_INITIAL_MAX_DIGITS;

  mpz_clear(mpz);

  mpz->sign = 1;
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

  mpz->sign = (z < 0) ? -1 : 1;

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

  for (i = 0; i < num_digits; i++) {
    /* parse the string backwards (little endian order) */
    mpz->digits[i] = digit_fromchar(str[num_digits - i - 1]);
  }

}

/**
 * @breif Destroy the mpz_t struct.
 */
__device__ __host__ void mpz_destroy(mpz_t *mpz) {
  free(mpz->digits);
  mpz->max_digits = 0;
}

/**
 * @brief Add two multiple precision integers.
 *
 *      dst := src1 + src2
 * 
 * @warning It is assumed that all mpz_t parameters have been initialized.
 * @warning Assumes that src1 and src2 are positive.
 */
__device__ __host__ void mpz_add(mpz_t *dst, mpz_t *src1, mpz_t *src2) {
  int src1_digit_count = mpz_count_digits(src1);
  int src2_digit_count = mpz_count_digits(src2);

  /* In addition, if the operand with the most digits has D digits, then
   * the result of the addition will have at most D + 1 digits. */
  int max_digits = max(src1_digit_count, src2_digit_count) + 1;

  /* Make sure all of the mpz structs have enough memory to hold all of
   * the digits. We will be doing 10's complement so everyone needs to 
   * have enough digits. */
  mpz_ensure_mem(dst,  max_digits);
  mpz_ensure_mem(src1, max_digits);
  mpz_ensure_mem(src2, max_digits);

  mpz_clear(dst);

  /* If both are negative, treate them as positive and negate the result */
  if (mpz_is_negative(src1) && mpz_is_negative(src2)) {
    digits_add(dst->digits, dst->max_digits, 
               src1->digits, src1->max_digits,
               src2->digits, src2->max_digits);
    dst->sign = -1;
  }
  else {
    digit_t carry_out;

    /* Perform 10's complement on negative numbers before adding */
    if (mpz_is_negative(src1)) digits_complement(src1->digits, src1->max_digits);
    if (mpz_is_negative(src2)) digits_complement(src2->digits, src2->max_digits);

    carry_out = digits_add(dst->digits, dst->max_digits, 
                           src1->digits, src1->max_digits,
                           src2->digits, src2->max_digits);
    
    if (carry_out == 0 && (mpz_is_negative(src1) || mpz_is_negative(src2))) {
      digits_complement(dst->digits, dst->max_digits);
      dst->sign = -1;
    }
    else {
      dst->sign = 1;
    }

    /* Undo the 10s complement after adding */
    if (mpz_is_negative(src1)) digits_complement(src1->digits, src1->max_digits);
    if (mpz_is_negative(src2)) digits_complement(src2->digits, src2->max_digits);
  }

}

/**
 * @breif Perform dst := src1 - src2.
 */
__device__ __host__ void mpz_sub(mpz_t *dst, mpz_t *src1, mpz_t *src2) {
  src2->sign *= -1;
  mpz_add(dst, src1, src2);
  src2->sign *= -1;
}

/**
 * @breif Return true if the the two mpz_t struct represent equivalent 
 * integers.
 */
__device__ __host__ int mpz_equal(mpz_t *a, mpz_t *b) {
  unsigned i;
  unsigned max_digits;
  int is_zero = 1;

  max_digits = max(a->max_digits, b->max_digits);

  for (i = 0; i < max_digits; i++) {
    digit_t ad = (i < a->max_digits) ? a->digits[i] : 0;
    digit_t bd = (i < b->max_digits) ? b->digits[i] : 0;

    is_zero = is_zero && (ad == 0);
    
    if (ad != bd) return 0;
  }

  /* Both numbers are either zero (in which case they are equal
   * and we don't care about the sign), or they have equivalent 
   * digits (in which case we just need to make sure their signs
   * match) */
  return (is_zero) || (a->sign == b->sign);
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

  (void) bufsize;

  if (buf != NULL) {
    str = buf;
  }
  else {
    str = (char *) malloc (sizeof(char) * (mpz->max_digits + 1));
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
