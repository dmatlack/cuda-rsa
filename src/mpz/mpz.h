/**
 * @file mpz.h
 *
 * @brief Multiple Precision Integer arithmetic.
 *
 * Most of the naming taken from the GNU Multiple Precision library.
 *
 * @author David Matlack (dmatlack)
 */
#ifndef __418_MPZ_H__
#define __418_MPZ_H__

#include "digit.h"

#define true 1
#define false 0

/* 
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 *      CURRENTLY ONLY SUPPORTING UNSIGNED INTEGERS
 *
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

/** @breif struct used to represent multiple precision integers (Z) */
typedef struct {
  digit_t     *digits;         // little endian order
  size_t       num_digits;     // number of digits in the array
} mpz_t;

void mpz_init
        (mpz_t *mpz);

void mpz_set_ui
        (mpz_t *mpz, unsigned int z);

void mpz_set_str
        (mpz_t *mpz, char *str);

void mpz_destroy
        (mpz_t *mpz);

void mpz_add
        (mpz_t *dest, mpz_t *source1, mpz_t *source2);

int  mpz_equal
        (mpz_t *a, mpz_t *b);

// To Implement:
//  mpz_sub
//  mpz_mult
//  mpz_div
//  mpz_lt
//  mpz_lte
//  mpz_gt
//  mpz_gte
//  mpz_mod

char* mpz_get_str
        (mpz_t *mpz);

#endif /* __418_MP_H__ */
