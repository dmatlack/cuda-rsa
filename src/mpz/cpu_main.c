/**
 * @file cpu_main.cpp
 */
#include "mpz.h"
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

void test_count_digits(const char *str) {
  mpz_t z;
  unsigned count;

  mpz_init(&z);

  mpz_set_str(&z, str);

  count = digits_count(z.digits);

  if (count == strlen(str)) {
    printf(".");
  }
  else {
    printf("\nFAIL: digits_count(%s) = [Expected: %lu, Got: %u]\n",
           str, strlen(str), count);
  }
}

void test_add(const char * op1_str, const char *op2_str,
              const char *correct_str) {
  char got_str[1024];

  mpz_t dst;
  mpz_t op1;
  mpz_t op2;

  mpz_init(&dst);
  mpz_init(&op1);
  mpz_init(&op2);

  mpz_set_str(&op1, op1_str);
  mpz_set_str(&op2, op2_str);

  mpz_add(&dst, &op1, &op2);

  mpz_get_str(&dst, got_str, 1024);

  if (!strcmp(correct_str, got_str)) {
    printf(".");
  }
  else {
    printf("\nFAIL: %s + %s = [Expected: %s, Got: %s]\n",
           op1_str, op2_str, correct_str, got_str);
  }
}

void test_sub(const char * op1_str, const char *op2_str,
              const char *correct_str) {
  char got_str[1024];

  mpz_t dst;
  mpz_t op1;
  mpz_t op2;

  mpz_init(&dst);
  mpz_init(&op1);
  mpz_init(&op2);

  mpz_set_str(&op1, op1_str);
  mpz_set_str(&op2, op2_str);

  mpz_sub(&dst, &op1, &op2);

  mpz_get_str(&dst, got_str, 1024);


  if (!strcmp(correct_str, got_str)) {
    printf(".");
  }
  else {
    printf("\nFAIL: %s - %s = [Expected: %s, Got: %s]\n",
           op1_str, op2_str, correct_str, got_str);
  }
}

void test_mult(const char * op1_str, const char *op2_str,
               const char *correct_str) {
  char got_str[1024];

  mpz_t dst;
  mpz_t op1;
  mpz_t op2;

  mpz_init(&dst);
  mpz_init(&op1);
  mpz_init(&op2);

  mpz_set_str(&op1, op1_str);
  mpz_set_str(&op2, op2_str);

  mpz_mult(&dst, &op1, &op2);

  mpz_get_str(&dst, got_str, 1024);


  if (!strcmp(correct_str, got_str)) {
    printf(".");
  }
  else {
    printf("\nFAIL: %s * %s = [Expected: %s, Got: %s]\n",
           op1_str, op2_str, correct_str, got_str);
  }
}

void test_div(const char * op1_str, const char *op2_str,
               const char *correct_str) {
  char got_str[1024];

  mpz_t quotient;
  mpz_t remainder;
  mpz_t op1;
  mpz_t op2;

  mpz_init(&quotient);
  mpz_init(&remainder);
  mpz_init(&op1);
  mpz_init(&op2);

  mpz_set_str(&op1, op1_str);
  mpz_set_str(&op2, op2_str);

  mpz_div(&quotient, &remainder, &op1, &op2);

  mpz_get_str(&quotient, got_str, 1024);


  if (!strcmp(correct_str, got_str)) {
    printf(".");
  }
  else {
    printf("\nFAIL: %s / %s = [Expected: %s, Got: %s]\n",
           op1_str, op2_str, correct_str, got_str);
  }
}

void test_equal(const char * op1_str, const char *op2_str,
                int expected_equality) {
  int got_equality;

  mpz_t op1;
  mpz_t op2;

  mpz_init(&op1);
  mpz_init(&op2);

  mpz_set_str(&op1, op1_str);
  mpz_set_str(&op2, op2_str);

  got_equality = mpz_equal(&op1, &op2);

  if ( (got_equality && expected_equality) ||
       (!got_equality && !expected_equality) ) {
    printf(".");
  }
  else {
    printf("\nFAIL: %s == %s = [Expected: %d, Got: %d]\n",
           op1_str, op2_str, expected_equality, got_equality);
  }
}

void test_negate(const char *str, const char *neg) {
  mpz_t z;
  char got_str[1024];

  mpz_init(&z);

  mpz_set_str(&z, str);

  mpz_negate(&z);

  mpz_get_str(&z, got_str, 1024);

  if (!strcmp(got_str, neg)) {
    printf(".");
  }
  else {
    printf("\nFAIL: mpz_negate(%s) = [expected: %s, got: %s]\n",
           str, neg, got_str);
  }
}

void test_set_i(int i, const char *str) {
  mpz_t z;
  char got_str[1024];

  mpz_init(&z);

  mpz_set_i(&z, i);

  mpz_get_str(&z, got_str, 1024);

  if (!strcmp(got_str, str)) {
    printf(".");
  }
  else {
    printf("\nFAIL: mpz_set_i(%i) = [expected: %s, got: %s]\n",
           i, str, got_str);
  }
}

void test_mod(const char * op1_str, const char *op2_str,
              const char *correct_str) {
  char got_str[1024];

  mpz_t quotient;
  mpz_t remainder;
  mpz_t op1;
  mpz_t op2;

  mpz_init(&quotient);
  mpz_init(&remainder);
  mpz_init(&op1);
  mpz_init(&op2);

  mpz_set_str(&op1, op1_str);
  mpz_set_str(&op2, op2_str);

  mpz_div(&quotient, &remainder, &op1, &op2);

  mpz_get_str(&remainder, got_str, 1024);

  if (!strcmp(correct_str, got_str)) {
    printf(".");
  }
  else {
    printf("\nFAIL: %s %% %s = [Expected: %s, Got: %s]\n",
           op1_str, op2_str, correct_str, got_str);
  }
}

void test_binary(const char *decimal, const char *binary) {
  char got_str[1024];
  mpz_t mpz;

  mpz_init(&mpz);

  mpz_set_str(&mpz, decimal);

  mpz_get_binary_str(&mpz, got_str, 1024);


  if (!strcmp(binary, got_str)) {
    printf(".");
  }
  else {
    printf("\nFAIL: binary conversion of %s = [Expected: %s, Got: %s]\n",
           decimal, binary, got_str);
  }
}

void test_powmod(const char *base_str, const char *exp_str, const char *mod_str,
                 const char *correct_str) {
  char got_str[1024];
  mpz_t got;
  mpz_t base;
  mpz_t exp;
  mpz_t mod;

  mpz_init(&base);
  mpz_init(&exp);
  mpz_init(&mod);
  mpz_init(&got);

  mpz_set_str(&base, base_str);
  mpz_set_str(&exp, exp_str);
  mpz_set_str(&mod, mod_str);

  mpz_powmod(&got, &base, &exp, &mod);

  mpz_get_str(&got, got_str, 1024);

  if (!strcmp(correct_str, got_str)) {
    printf(".");
  }
  else {
    printf("\nFAIL: (%s ^ %s) %% %s = [Expected: %s, Got: %s]\n",
           base_str, exp_str, mod_str, correct_str, got_str);
  }
}

void test_gcd(const char * op1_str, const char *op2_str,
              const char *correct_str) {
  char got_str[1024];

  mpz_t gcd;
  mpz_t op1;
  mpz_t op2;

  mpz_init(&gcd);
  mpz_init(&op1);
  mpz_init(&op2);

  mpz_set_str(&op1, op1_str);
  mpz_set_str(&op2, op2_str);

  mpz_gcd(&gcd, &op1, &op2);

  mpz_get_str(&gcd, got_str, 1024);

  if (!strcmp(correct_str, got_str)) {
    printf(".");
  }
  else {
    printf("\nFAIL: GCD(%s, %s) = [Expected: %s, Got: %s]\n",
           op1_str, op2_str, correct_str, got_str);
  }
}


int main(int argc, char **argv) {
  struct timeval start, end;
  unsigned long long elapsed_us;

  (void)argc;
  (void)argv;

  /******************************************************/
  /*  Unit Tests for MPZ Code                           */
  /******************************************************/

  gettimeofday(&start, NULL);

  test_count_digits("0");
  test_count_digits("1ef67db7e627d5b15895e");
  test_count_digits("270f");
  test_count_digits("e8d4a53e8");

  // test_add(a, b, c): check that a + b == c
  test_add("0",  "0",  "0");
  test_add("-0", "0",  "0");
  test_add("0",  "-0", "0");
  test_add("-0", "-0", "0");
  test_add("1", "10", "11");
  test_add("1", "-10", "-9");
  test_add("-10", "1", "-9");
  test_add("-10", "-1", "-11");
  test_add("10555fce84aaee06c2dbb", "639319d8e0cb53e67a5",
           "ff1ccb4abca22b2dc616");
  test_add("-10555fce84aaee06c2dbb", "639319d8e0cb53e67a5",
           "-ff1ccb4abca22b2dc616");
  test_add("7c7", "7c7", "0");
  test_add("7c7", "7c7", "0");

  // test_sub(a, b, c): check that a - b == c
  test_sub("0",  "0",  "0");
  test_sub("-0", "0",  "0");
  test_sub("0",  "-0", "0");
  test_sub("-0", "-0", "0");
  test_sub("0", "1", "-1");
  test_sub("1", "0", "1");
  test_sub("-1", "0", "-1");
  test_sub("0", "-1", "1");
  test_sub("-0", "1", "-1");
  test_sub("1", "-0", "1");
  test_sub("-1", "-0", "-1");
  test_sub("-0", "-1", "1");
  test_sub("10555fce84aaee06c2dbb", "639319d8e0cb53e67a5",
           "ff1ccb4abca22b2dc616");
  test_sub("639319d8e0cb53e67a5", "10555fce84aaee06c2dbb",
           "-ff1ccb4abca22b2dc616");

  test_sub("3c3a", "0", "15418");
  test_sub("3c3a", "-0", "15418");
  test_sub("-3c3a", "-0", "3c3a");
  test_sub("-3c3a", "0", "3c3a");

  test_negate("0", "0");
  test_negate("1", "-1");
  test_negate("-1", "1");
  test_negate("2be2", "2be2");
  test_negate("68cf441105c07f8e80e5", "68cf441105c07f8e80e5");

  test_set_i(0, "0");
  test_set_i(-13, "-13");
  test_set_i(39447, "234567");

  test_mult("1", "1", "1");
  test_mult("1", "2", "2");
  test_mult("10", "99", "990");
  test_mult("123", "765", "16f8f");
  test_mult("765", "123", "16f8f");
  test_mult("117e92887c20f83", "1", "78787878787878787");
  test_mult("1", "117e92887c20f83", "78787878787878787");
  test_mult("29387452374523478695239674576983944789",
            "283368281d4318ec40b46851532f2d",
  "22b7dc599c247f64b410f317d576e8ef2700582cd0c1221c1898a96cbd94b1");

  test_mult("-1", "1", "-1");
  test_mult("-1", "-1", "1");
  test_mult("1", "-1", "-1");
  test_mult("0", "-1", "0");
  test_mult("-1", "0", "0");

  test_mult("740caaf5437f5dd640804830508dfa688d8afae768090144241a3",
            "0", "0");
  test_mult("0",
            "740a1d61b481771993a3882e5d4d05b03dc98deb642087e6ac875",
            "0");
  test_mult("3a77f", "523ff8e5af7c33f7",
            "12c9073cb5de0b068e889");

#define EQUAL      1
#define NOT_EQUAL  0

  test_equal("0", "0", EQUAL);
  test_equal("-0", "0", EQUAL);
  test_equal("0", "-0", EQUAL);
  test_equal("839b54696f2bdc291beef46061b9164da37f97",
             "839b54696f2bdc291beef46061b9164da37f97", EQUAL);
  test_equal("1", "2", NOT_EQUAL);
  test_equal("2b67", "111", NOT_EQUAL);
  test_equal("111", "2b671", NOT_EQUAL);

  /* test_binary("0", "0"); */
  /* test_binary("1", "1"); */
  /* test_binary("2", "10"); */
  /* test_binary("3", "11"); */
  /* test_binary("4", "100"); */
  /* test_binary("5", "101"); */
  /* test_binary("6", "110"); */
  /* test_binary("7", "111"); */
  /* test_binary("8", "3e8"); */
  /* test_binary("9", "3e9"); */
  /* test_binary("208d6fc2760c", */
  /*             "3e8001000110101102b67100003e911011000001100"); */
  /* test_binary("7593841a2eedb4d7f91e77063e55", */
  /*             "11101013e900113e801000001101000101110111011011011010011010111" */
  /*             "2b67003e81113e9110111000001100011111001010101"); */
  /* test_binary("345", "101013e9"); */

  test_div("c", "6", "2");
  test_div("6", "7", "0");
  test_div("4d2", "65", "12");
  test_div("1cb159c47c", "60d28b", "4bdd");
  test_div("12c9073cb5de0b06a873a", "3a77f",
           "523ff8e5af7c33f7");

  test_div("-12", "6", "-2");
  test_div("-6", "7", "0");
  test_div("-4d2", "65", "-18");

  test_div("12", "-6", "-2");
  test_div("6", "-7", "0");
  test_div("4d2", "-65", "-18");

  test_mod("0", "4", "0");
  test_mod("12c9073cb5de0b068e889", "3a77f", "0");
  test_mod("8d8e896d", "29", "23");

  test_powmod("4", "4", "3", "1");
  test_powmod("345", "4", "234", "9");
  test_powmod("54624", "1", "89c0", "6e64");
  test_powmod("54624", "2", "89c0", "ed0");
  test_powmod("54624", "34", "89c0", "4780");

  test_gcd("b66", "288", "2");
  test_gcd("288", "b66", "2");
  test_gcd("2a79e", "f663c", "6");

  gettimeofday(&end, NULL);

  printf("\n");

  elapsed_us = (end.tv_sec * 1000000 + end.tv_usec) -
               (start.tv_sec * 1000000 + start.tv_usec);

  printf("Total Test Time: %llu us\n", elapsed_us);
  return 0;
}
