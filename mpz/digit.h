/**
 * @file digit.c
 *
 * @brief Library of functions that operate on arrays of digits.
 *
 * Arrays of digits are assumed to be in little endian order.
 *
 * @author David Matlack (dmatlack)
 */
#ifndef __418_DIGIT_H__
#define __418_DIGIT_H__

#ifndef __CUDACC__ /* when compiling with gcc... */

#define __device__
#define __host__

#include <string.h>

#endif /* __CUDACC__ */

typedef unsigned char digit_t;

#define DIGIT_BASE 10

/**
 * @brief Return true (non-zero) if all of the digits in the digits array
 * are zero (and thus the corresponding number is zero.
 */
__device__ __host__ int digits_is_zero(digit_t *digits, 
                                       unsigned num_digits) {
  unsigned i;

  for (i = 0; i < num_digits; i++) {
    if (digits[i] != 0) return 0;
  }
  return 1;
}

/**
 * @brief Find the result of a + b + carry. Store the resulting carry of this
 * operation back in the carry pointer.
 */
__device__ __host__ digit_t digit_add(digit_t a, digit_t b, 
                                      digit_t *carry) {
  digit_t result = a + b + *carry;

  if (result >= DIGIT_BASE) {
    *carry = result / DIGIT_BASE;
    result = result % DIGIT_BASE;
  }
  else {
    *carry = 0;
  }
  
  return result;
}

__device__ __host__ void digits_complement(digit_t *digits, 
                                           unsigned num_digits) {
  digit_t carry = 0;
  unsigned i;

  // Complement each digit
  for (i = 0; i < num_digits; i++) {
    digits[i] = (DIGIT_BASE - 1) - digits[i];
  }

  // Add 1
  i = 0;
  carry = 1;
  while (carry != 0) {
    digits[i] = digit_add(digits[i], 0, &carry);
    i++;
  }

}

__device__ __host__ digit_t digits_add(digit_t *sum, unsigned sum_num_digits, 
                                       digit_t *op1, unsigned op1_num_digits,
                                       digit_t *op2, unsigned op2_num_digits) {
  digit_t carry = 0;
  unsigned i;

  for (i = 0; i < sum_num_digits; i++) {
    digit_t a = (i < op1_num_digits) ? op1[i] : 0;
    digit_t b = (i < op2_num_digits) ? op2[i] : 0;

    sum[i] = digit_add(a, b, &carry);
  }

  return carry;
}

__device__ __host__ char digit_tochar(digit_t d) {
  return '0' + d;
}

__device__ __host__ digit_t digit_fromchar(char c) {
  if (c < '0' || c > '9') {
    c = '0';
  }

  return c - '0';
}

#endif /* __418_DIGIT_H__ */
