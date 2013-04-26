#include <stdio.h>

int main(int argc, char *argv[]) {
  (void) argc;
  (void) argv;

  printf("unsigned: %u bytes\n", sizeof(unsigned));
  printf("unsigned long: %u bytes\n", sizeof(unsigned long));
  printf("unsigned long long: %u bytes\n", sizeof(unsigned long long));

  return 0;
}
