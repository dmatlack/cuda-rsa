/**
 * @file mpz.c
 *
 * @brief Multiple Precision arithmetic code.
 *
 * @author David Matlack (dmatlack)
 */
#include "mpz.h"

#include <stdio.h>
#include <stdlib.h>

static inline int min(int a, int b) { return (a < b) ? a : b; }
static inline int max(int a, int b) { return (a > b) ? a : b; }

/** @brief Called if the program runs out of memory */
static void mpz_memory_error(char *function) {
  fprintf(stderr, "MP: Ran out of memory during call to %s."
                  " Terminating.\n", function);
  exit(-4);
}

/**
 * @brief Check that the given mpz_t struct has room for num_digits. If not,
 * allocate memory for enough digits.
 */
static void mpz_ensure_mem(mpz_t *mpz, size_t num_digits) {
  if (mpz->num_digits < num_digits) {
    if (mpz->num_digits > 0) free(mpz->digits);

    mpz->digits = (digit_t *) calloc (num_digits, sizeof(digit_t));
    mpz->num_digits = num_digits;

    if (NULL == mpz->digits) mpz_memory_error("mpz_alloc");
  }
}

/**
 * @brief Intialize the mpz_t struct with no intial value.
 */
void mpz_init(mpz_t *mpz) {
  mpz->num_digits = 0;
}

/**
 * @brief Set the mpz integer to the provided unsigned integer.
 *
 * @warning Assumes the mpz struct has been initialized.
 */
void mpz_set_ui(mpz_t *mpz, unsigned int z) {
  size_t num_digits = 64;

  mpz_ensure_mem(mpz, num_digits);
  mpz->num_digits = num_digits;

  digits_parse(mpz->digits, mpz->num_digits, z);
}

/**
 * @brief Set the mpz integer based on the provided string.
 *
 * @warning Assumes the mpz struct has been initialized.
 */
void mpz_set_str(mpz_t *mpz, char *str) {
  size_t num_digits = strlen(str);
  size_t i;

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
void mpz_destroy(mpz_t *mpz) {
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
void mpz_add(mpz_t *dst, mpz_t *src1, mpz_t *src2) {
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
int mpz_equal(mpz_t *a, mpz_t *b) {
  size_t i;
  size_t min_digits;

  min_digits = min(a->num_digits, b->num_digits);

  for (i = 0; i < min_digits; i++) {
    digit_t ad = (i < a->num_digits) ? a->digits[i] : 0;
    digit_t bd = (i < b->num_digits) ? b->digits[i] : 0;
    
    if (ad != bd) return false;
  }

  return true;
}

/**
 * @breif Return the string representation of the integer represented by the
 * mpz_t struct.
 *
 * @warning The string is dynamically allocated and must therefore be freed by 
 * the user.
 */
char* mpz_get_str(mpz_t *mpz) {
  char *str = (char *) malloc (sizeof(char) * (mpz->num_digits + 1));
  int print_zeroes = 0; // don't print leading 0s
  int i, str_index = 0;

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
