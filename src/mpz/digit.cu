/**
 * @file digit.c
 *
 * @brief Library of functions that operate on arrays of digits.
 *
 * Arrays of digits are assumed to be in little endian order.
 *
 * @author David Matlack (dmatlack)
 */

typedef unsigned char digit_t;

#define DIGIT_BASE 10

/**
 * @brief Return true (non-zero) if all of the digits in the digits array
 * are zero (and thus the corresponding number is zero.
 */
__device__ int digits_is_zero(digit_t *digits, unsigned num_digits) {
  unsigned i;

  for (i = 0; i < num_digits; i++) {
    if (digits[i] != 0) return false;
  }
  return true;
}

/**
 * @brief Parse an unsigned int into an array of digits.
 */
__device__ void digits_parse(digit_t *digits, unsigned num_digits, unsigned ui) {
  unsigned i;
  
  memset(digits, 0, num_digits * sizeof(digit_t));

  i = 0;
  while (ui > 0 && i < num_digits) {
    digits[i++] = ui % 10;
    ui /= 10;
  }
}

/**
 * @brief Find the result of a + b + carry. Store the resulting carry of this
 * operation back in the carry pointer.
 */
__device__ __inline__ digit_t digit_add(digit_t a, digit_t b, digit_t *carry) {
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

__device__ void digits_add(digit_t *sum, unsigned sum_num_digits, 
                           digit_t *op1, unsigned op1_num_digits,
                           digit_t *op2, unsigned op2_num_digits) {
  digit_t carry = 0;
  unsigned i;

  for (i = 0; i < sum_num_digits; i++) {
    digit_t a = (i < op1_num_digits) ? op1[i] : 0;
    digit_t b = (i < op2_num_digits) ? op2[i] : 0;

    sum[i] = digit_add(a, b, &carry);
  }
}

__device__ char digit_tochar(digit_t d) {
  return '0' + d;
}

__device__ digit_t digit_fromchar(char c) {
  if (c < '0' || c > '9') {
    c = '0';
  }

  return c - '0';
}
