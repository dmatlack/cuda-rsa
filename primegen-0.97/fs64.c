#include "fs64.h"

unsigned int scan_uint64(char *s,uint64 *u)
{
  unsigned int pos = 0;
  uint64 result = 0;
  uint64 c;
  while ((c = (uint64) (unsigned char) (s[pos] - '0')) < 10) {
    result = result * 10 + c;
    ++pos;
  }
  *u = result;
  return pos;
}

unsigned int fmt_uint64(char *s,uint64 u)
{
  unsigned int len = 1;
  uint64 q = u;
  while (q > 9) { ++len; q /= 10; }
  if (s) { s += len; do { *--s = '0' + (u % 10); u /= 10; } while(u); }
  return len;
}
