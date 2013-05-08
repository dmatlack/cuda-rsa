#include "auto_home.h"

void hier()
{
  h(auto_home,-1,-1,0755);

  d(auto_home,"bin",-1,-1,0755);
  d(auto_home,"lib",-1,-1,0755);
  d(auto_home,"include",-1,-1,0755);
  d(auto_home,"man",-1,-1,0755);
  d(auto_home,"man/man1",-1,-1,0755);
  d(auto_home,"man/cat1",-1,-1,0755);
  d(auto_home,"man/man3",-1,-1,0755);
  d(auto_home,"man/cat3",-1,-1,0755);

  c(auto_home,"include","primegen.h",-1,-1,0644);
  c(auto_home,"lib","primegen.a",-1,-1,0644);
  c(auto_home,"bin","primes",-1,-1,0755);
  c(auto_home,"bin","primegaps",-1,-1,0755);

  c(auto_home,"man/man1","primes.1",-1,-1,0644);
  c(auto_home,"man/man1","primegaps.1",-1,-1,0644);
  c(auto_home,"man/man3","primegen.3",-1,-1,0644);

  c(auto_home,"man/cat1","primes.0",-1,-1,0644);
  c(auto_home,"man/cat1","primegaps.0",-1,-1,0644);
  c(auto_home,"man/cat3","primegen.0",-1,-1,0644);
}
