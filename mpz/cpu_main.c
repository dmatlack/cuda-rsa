/**
 * @file cpu_main.cpp
 */
#include "mpz.h"
#include <string.h>
#include <stdio.h>

void test_count_digits(const char *str) {
  mpz_t z;
  unsigned count;

  mpz_init(&z);

  mpz_set_str(&z, str);

  count = mpz_count_digits(&z);

  if (count == strlen(str)) {
    printf(".");
  }
  else {
    printf("\nFAIL: mpz_count_digits(%s) = [Expected: %lu, Got: %u]\n",
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

int main(int argc, char **argv) {

  (void)argc;
  (void)argv;

  /******************************************************/
  /*  Unit Tests for MPZ Code                           */
  /******************************************************/

  test_count_digits("0");
  test_count_digits("2339487239847298374928734");
  test_count_digits("9999");
  test_count_digits("1000000000000");

  // test_add(a, b, c): check that a + b == c
  test_add("0",  "0",  "0");
  test_add("-0", "0",  "0");
  test_add("0",  "-0", "0");
  test_add("-0", "-0", "0");
  test_add("1", "10", "11");
  test_add("1", "-10", "-9");
  test_add("-10", "1", "-9");
  test_add("-10", "-1", "-11");

  test_add("1234123849173249817324987", "-29389238479283749283749",
           "1204734610693966068041238");
  test_add("-1234123849173249817324987", "29389238479283749283749",
           "-1204734610693966068041238");

  // test_sub(a, b, c): check that a - b == c
  test_sub("0",  "0",  "0");
  test_sub("-0", "0",  "0");
  test_sub("0",  "-0", "0");
  test_sub("-0", "-0", "0");

  test_sub("1234123849173249817324987", "29389238479283749283749",
           "1204734610693966068041238");
  test_sub("29389238479283749283749", "1234123849173249817324987", 
           "-1204734610693966068041238");
  
  printf("\n");
  return 0;
}
