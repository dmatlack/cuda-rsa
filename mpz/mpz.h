/**
 * @file mpz.c
 *
 * @brief Multiple Precision arithmetic code.
 *
 * @author David Matlack (dmatlack)
 */
#ifndef __418_MPZ_H__
#define __418_MPZ_H__

#include "cuda_string.h"
#include "digit.h"

/** @breif struct used to represent multiple precision integers (Z) */
typedef struct {
  digit_t     *digits;         // little endian order
  unsigned     num_digits;     // number of digits in the array
} mpz_t;

/** @brief Called if the program runs out of memory */
__device__ __host__ void mpz_memory_error(char *function) {
  // TODO
}

/**
 * @brief Check that the given mpz_t struct has room for num_digits. If not,
 * allocate memory for enough digits.
 */
__device__ __host__ void mpz_ensure_mem(mpz_t *mpz, unsigned num_digits) {
  int i;
  
  if (mpz->num_digits < num_digits) {
    if (mpz->num_digits > 0) free(mpz->digits);

    mpz->digits = (digit_t *) malloc (num_digits * sizeof(digit_t));
    mpz->num_digits = num_digits;

    for (i = 0; i < num_digits; i++) {
      mpz->digits[i] = 0;
    }
    

    if (NULL == mpz->digits) mpz_memory_error("mpz_alloc");
  }
}

/**
 * @brief Intialize the mpz_t struct with no intial value.
 */
__device__ __host__ void mpz_init(mpz_t *mpz) {
  mpz->num_digits = 0;
}

/**
 * @brief Set the mpz integer to the provided unsigned integer.
 *
 * @warning Assumes the mpz struct has been initialized.
 */
__device__ __host__ void mpz_set_ui(mpz_t *mpz, unsigned int z) {
  unsigned num_digits = 64;

  mpz_ensure_mem(mpz, num_digits);
  mpz->num_digits = num_digits;

  digits_parse(mpz->digits, mpz->num_digits, z);
}

/**
 * @brief Set the mpz integer based on the provided string.
 *
 * @warning Assumes the mpz struct has been initialized.
 */
__device__ __host__ void mpz_set_str(mpz_t *mpz, char *str) {
  unsigned num_digits = cuda_strlen(str);
  unsigned i;

  mpz_ensure_mem(mpz, num_digits);

  for (i = 0; i < num_digits; i++) {
    /* parse the string backwards (little endian order) */
    mpz->digits[i] = digit_fromchar(str[num_digits - i - 1]);
  }

  mpz->num_digits = num_digits;
}

/**
 * @breif Destroy the mpz_t struct.
 */
__device__ __host__ void mpz_destroy(mpz_t *mpz) {
  free(mpz->digits);
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
  int max_digits = max(src1->num_digits, src2->num_digits);

  mpz_ensure_mem(dst, max_digits);

  digits_add(dst->digits, dst->num_digits, 
             src1->digits, src1->num_digits,
             src2->digits, src2->num_digits);
}

/**
 * @breif Return true if the the two mpz_t struct represent equivalent 
 * integers.
 */
__device__ __host__ int mpz_equal(mpz_t *a, mpz_t *b) {
  unsigned i;
  unsigned min_digits;

  min_digits = min(a->num_digits, b->num_digits);

  for (i = 0; i < min_digits; i++) {
    digit_t ad = (i < a->num_digits) ? a->digits[i] : 0;
    digit_t bd = (i < b->num_digits) ? b->digits[i] : 0;
    
    if (ad != bd) return 0;
  }

  return 1;
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

  if (buf != NULL) {
    str = buf;
  }
  else {
    str = (char *) malloc (sizeof(char) * (mpz->num_digits + 1));
  }

  for (i = mpz->num_digits - 1; i >= 0; i--) {
    int digit = mpz->digits[i];

    if (digit != 0 || print_zeroes) {
      print_zeroes = 1;
      str[str_index++] = digit_tochar(digit);
    }
  }

  str[str_index] = (char) 0;

  return str;
}

#endif /* __418_MPZ_H__ */
