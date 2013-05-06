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

#ifndef __CUDACC__
#define MPZ_ENSURE_MEM(__mpz, __capacity) \
  do {\
    if ((__mpz)->capacity < (__capacity)) {\
      printf("MPZ memory error: %s at line %d.\n", __func__, __LINE__);\
      printf("\tmpz capacity: %u, requested capacity %u\n", (__mpz)->capacity, (__capacity));\
      exit(4);\
    }\
  } while (0) 
#else
#define MPZ_ENSURE_MEM(m,c)
#endif

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
  int is_leading_zero = true;
  int count = 0;
  int i;

  for (i = mpz->capacity - 1; i >= 0; i--) {
    digit_t d = mpz->digits[i];

    if (0 == d && is_leading_zero) continue;

    is_leading_zero = false;
    count++;
  }

  /* special case where all digits are 0 */
  if (count == 0) return 1;

  return count;
}

inline __device__ __host__ int mpz_is_negative(mpz_t *mpz) {
  if (digits_is_zero(mpz->digits, mpz->capacity)) return false;
  return (mpz->sign == -1);
}

inline __device__ __host__ void mpz_negate(mpz_t *mpz) {
  mpz->sign *= -1;
}

/**
 * @brief Set the mpz_t struct to zero.
 */
__device__ __host__ void mpz_clear(mpz_t *mpz) {
  unsigned i;

  for (i = 0; i < mpz->capacity; i++) {
    mpz->digits[i] = 0;
  }
}

/**
 * @brief If the number is zero, the sign should be zero.
 */
static inline __device__ __host__ void _assert_sign(mpz_t *mpz) {
#ifndef __CUDACC__
  if (digits_is_zero(mpz->digits, mpz->capacity)) {
    assert(0 == mpz->sign);
  }
  else {
    assert(1 == mpz->sign || -1 == mpz->sign);
  }
#endif
}

/**
 * @brief Initialize an mpz_t struct. Essentially zeroing out the digits array.
 */
__device__ __host__ void mpz_init(mpz_t *mpz) {
  mpz->capacity = DIGITS_CAPACITY;
  mpz_clear(mpz);
  mpz->sign = 0;
}

/**
 * @brief Assign an mpz_t struct to the value of another mpz_t struct.
 */
__device__ __host__ void mpz_set(mpz_t *to, mpz_t *from) {
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
__device__ __host__ void mpz_set_i(mpz_t *mpz, int z) {
  unsigned i;

  mpz->sign = (z == 0) ? 0 : ((z > 0) ? 1: -1);

  z = abs(z);
  i = 0;
  for (i = 0; i < mpz->capacity; i++) {
    mpz->digits[i] = z % 10;
    z /= 10;
  }
}

/**
 * @brief Set the mpz integer based on the provided string.
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
  MPZ_ENSURE_MEM(mpz, num_digits);

  mpz_clear(mpz);

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
__device__ __host__ void mpz_destroy(mpz_t *mpz) {
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
  unsigned capacity = max(op1_digit_count, op2_digit_count) + 1;

  /* Make sure all of the mpz structs have enough memory to hold all of
   * the digits. We will be doing 10's complement so everyone needs to 
   * have enough digits. */
  MPZ_ENSURE_MEM(dst, capacity);
  MPZ_ENSURE_MEM(op1, capacity);
  MPZ_ENSURE_MEM(op2, capacity);

  mpz_clear(dst);

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
  unsigned capacity = max(op1_digit_count, op2_digit_count);

  /* In multiplication, if the operand with the most digits has D digits, 
   * then the result of the addition will have at most 2D digits. */
  MPZ_ENSURE_MEM(dst, 2*capacity);
  MPZ_ENSURE_MEM(op1,   capacity);
  MPZ_ENSURE_MEM(op2,   capacity);

  /* We pass in capacity as the number of digits rather that the actual
   * number of digits in each mpz_t struct. This is done because the 
   * multiplication code has some assumptions and optimizations (e.g.
   * op1 and op2 to have the same number of digits) */
  digits_mult(dst->digits, op1->digits, op2->digits, capacity);

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

__device__ __host__ void mpz_print(mpz_t *mpz) {
  char str[1024];

  mpz_get_str(mpz, str, 1024);
  printf("%s", str);
}

__device__ __host__ char* mpz_get_binary_str(mpz_t *mpz, char *buf, 
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
__device__ __host__ void mpz_div(mpz_t *q, mpz_t *r, mpz_t *n, mpz_t *d) {
  unsigned n_digit_count = mpz_count_digits(n);
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

  n->sign = 1;
  d->sign = 1;

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
    q->sign = 0;
    mpz_set(r, n);
  }

  n->sign = nsign;
  d->sign = dsign;
}

#endif /* __418_MPZ_H__ */
