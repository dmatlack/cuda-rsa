/**
 * @file digit.h
 *
 * @breif Functions that operate on digits and arrays of digits.
 *
 * @author David Matlack
 */
#ifndef __418_DIGIT_H__
#define __418_DIGIT_H__

#include <string.h>

typedef unsigned char digit_t;

#define DIGIT_BASE 10

#define true 1
#define false 0

int digits_is_zero
       (digit_t *digits, size_t num_digits);

void digits_parse
       (digit_t *digits, size_t num_digits, unsigned ui);

void digits_add
       (digit_t *sum, size_t sum_num_digits, 
        digit_t *op1, size_t op1_num_digits,
        digit_t *op2, size_t op2_num_digits);

char digit_tochar
       (digit_t d);

digit_t  digit_fromchar
            (char c);


#endif /* __418_DIGIT_H__ */
