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
  test_add("1991", "-1991", "0");
  test_add("-1991", "1991", "0");

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
  test_sub("1234123849173249817324987", "29389238479283749283749",
           "1204734610693966068041238");
  test_sub("29389238479283749283749", "1234123849173249817324987", 
           "-1204734610693966068041238");

  test_sub("15418", "0", "15418");
  test_sub("15418", "-0", "15418");
  test_sub("-15418", "-0", "-15418");
  test_sub("-15418", "0", "-15418");

  test_negate("0", "0");
  test_negate("1", "-1");
  test_negate("-1", "1");
  test_negate("11234", "-11234");
  test_negate("494949494949494494494949", "-494949494949494494494949");

  test_set_i(0, "0");
  test_set_i(-13, "-13");
  test_set_i(234567, "234567");

  test_mult("1", "1", "1");
  test_mult("1", "2", "2");
  test_mult("10", "99", "990");
  test_mult("123", "765", "94095");
  test_mult("765", "123", "94095");
  test_mult("78787878787878787", "1", "78787878787878787");
  test_mult("1", "78787878787878787", "78787878787878787");
  test_mult("293874523745234786952396745769823456789",
            "208734529374856923746592873465982765",
  "61341760409221799187557653420334856048688516904554850524715252753396241585");

  test_mult("-1", "1", "-1");
  test_mult("-1", "-1", "1");
  test_mult("1", "-1", "-1");
  test_mult("0", "-1", "0");
  test_mult("-1", "0", "0");

  test_mult("2983749283749287349827394872398479283749827394872938479283749283",
            "0", "0");
  test_mult("0",
            "2983492873401874018273482347156347856101573456091873465093245045",
            "0");
  test_mult("239487", "5926729300018213879", 
            "1419374619873461987240073");

#define EQUAL      1
#define NOT_EQUAL  0

  test_equal("0", "0", EQUAL);
  test_equal("-0", "0", EQUAL);
  test_equal("0", "-0", EQUAL);
  test_equal("2934928749191929923847234234234234293847293847", 
             "2934928749191929923847234234234234293847293847", EQUAL);
  test_equal("1", "2", NOT_EQUAL);
  test_equal("11111", "111", NOT_EQUAL);
  test_equal("111", "111111", NOT_EQUAL);

  test_binary("0", "0");
  test_binary("1", "1");
  test_binary("2", "10");
  test_binary("3", "11");
  test_binary("4", "100");
  test_binary("5", "101");
  test_binary("6", "110");
  test_binary("7", "111");
  test_binary("8", "1000");
  test_binary("9", "1001");
  test_binary("35791837492748", 
              "1000001000110101101111110000100111011000001100");
  test_binary("2384729347191823792472938479238741", 
              "111010110010011100001000001101000101110111011011011010011010111"
              "111110010001111001110111000001100011111001010101");
  test_binary("345", "101011001");

  test_div("12", "6", "2");
  test_div("6", "7", "0");
  test_div("1234", "65", "18");
  test_div("123234534524", "6345355", "19421");
  test_div("1419374619873461987346234", "239487",
           "5926729300018213879");

  test_div("-12", "6", "-2");
  test_div("-6", "7", "0");
  test_div("-1234", "65", "-18");

  test_div("12", "-6", "-2");
  test_div("6", "-7", "0");
  test_div("1234", "-65", "-18");

  test_mod("0", "4", "0");
  test_mod("1419374619873461987240073", "239487", "0");
  test_mod("2374928749", "29", "23");

  test_powmod("4", "4", "3", "1");
  test_powmod("345", "4", "234", "9");
  test_powmod("345636", "1", "35264", "28260");
  test_powmod("345636", "2", "35264", "3792");
  test_powmod("345636", "34", "35264", "18304");

  test_gcd("2918", "288", "2");
  test_gcd("288", "2918", "2");
  test_gcd("173982", "1009212", "6");

  gettimeofday(&end, NULL);

  printf("\n");

  elapsed_us = (end.tv_sec * 1000000 + end.tv_usec) - 
               (start.tv_sec * 1000000 + start.tv_usec);

  printf("Total Test Time: %llu us\n", elapsed_us);
  return 0;
}
