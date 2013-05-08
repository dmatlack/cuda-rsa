# Don't edit Makefile! Use conf-* for configuration.

SHELL=/bin/sh

default: it

auto-ccld.sh: \
conf-cc conf-ld warn-auto.sh
	( cat warn-auto.sh; \
	echo CC=\'`head -1 conf-cc`\'; \
	echo LD=\'`head -1 conf-ld`\' \
	) > auto-ccld.sh

auto-str: \
load auto-str.o substdio.a error.a str.a
	./load auto-str substdio.a error.a str.a 

auto-str.o: \
compile auto-str.c substdio.h readwrite.h exit.h
	./compile auto-str.c

auto_home.c: \
auto-str conf-home
	./auto-str auto_home `head -1 conf-home` > auto_home.c

auto_home.o: \
compile auto_home.c
	./compile auto_home.c

byte_copy.o: \
compile byte_copy.c byte.h
	./compile byte_copy.c

byte_cr.o: \
compile byte_cr.c byte.h
	./compile byte_cr.c

check: \
it instcheck
	./instcheck

compile: \
make-compile warn-auto.sh systype
	( cat warn-auto.sh; ./make-compile "`cat systype`" ) > \
	compile
	chmod 755 compile

eratspeed: \
load eratspeed.o
	./load eratspeed 

eratspeed.o: \
compile eratspeed.c timing.h hasrdtsc.h hasgethr.h uint32.h
	./compile eratspeed.c

error.a: \
makelib error.o error_str.o
	./makelib error.a error.o error_str.o

error.o: \
compile error.c error.h
	./compile error.c

error_str.o: \
compile error_str.c error.h
	./compile error_str.c

find-systype: \
find-systype.sh auto-ccld.sh
	cat auto-ccld.sh find-systype.sh > find-systype
	chmod 755 find-systype

fs64.o: \
compile fs64.c fs64.h uint64.h
	./compile fs64.c

hasgethr.h: \
trygethr.c compile load
	( ( ./compile trygethr.c && ./load trygethr ) >/dev/null \
	2>&1 \
	&& echo \#define HASGETHRTIME 1 || exit 0 ) > hasgethr.h
	rm -f trygethr.o

hasrdtsc.h: \
tryrdtsc.c compile load
	( ( ./compile tryrdtsc.c && ./load tryrdtsc && ./tryrdtsc \
	) >/dev/null 2>&1 \
	&& echo \#define HASRDTSC 1 || exit 0 ) > hasrdtsc.h
	rm -f tryrdtsc.o tryrdtsc

hier.o: \
compile hier.c auto_home.h
	./compile hier.c

install: \
load install.o hier.o auto_home.o strerr.a substdio.a open.a error.a \
str.a
	./load install hier.o auto_home.o strerr.a substdio.a \
	open.a error.a str.a 

install.o: \
compile install.c substdio.h strerr.h error.h open.h readwrite.h \
exit.h
	./compile install.c

instcheck: \
load instcheck.o hier.o auto_home.o strerr.a substdio.a error.a str.a
	./load instcheck hier.o auto_home.o strerr.a substdio.a \
	error.a str.a 

instcheck.o: \
compile instcheck.c strerr.h error.h readwrite.h exit.h
	./compile instcheck.c

int64.h: \
trylong64.c compile load int64.h1 int64.h2
	( ( ./compile trylong64.c && ./load trylong64 && \
	./trylong64 ) >/dev/null 2>&1 \
	&& cat int64.h1 || cat int64.h2 ) > int64.h
	rm -f trylong64.o trylong64

it: \
man prog

load: \
make-load warn-auto.sh systype
	( cat warn-auto.sh; ./make-load "`cat systype`" ) > load
	chmod 755 load

make-compile: \
make-compile.sh auto-ccld.sh
	cat auto-ccld.sh make-compile.sh > make-compile
	chmod 755 make-compile

make-load: \
make-load.sh auto-ccld.sh
	cat auto-ccld.sh make-load.sh > make-load
	chmod 755 make-load

make-makelib: \
make-makelib.sh auto-ccld.sh
	cat auto-ccld.sh make-makelib.sh > make-makelib
	chmod 755 make-makelib

makelib: \
make-makelib warn-auto.sh systype
	( cat warn-auto.sh; ./make-makelib "`cat systype`" ) > \
	makelib
	chmod 755 makelib

man: \
primes.0 primespeed.0 primegaps.0 primegen.0

open.a: \
makelib open_read.o open_trunc.o
	./makelib open.a open_read.o open_trunc.o

open_read.o: \
compile open_read.c open.h
	./compile open_read.c

open_trunc.o: \
compile open_trunc.c open.h
	./compile open_trunc.c

primegaps: \
load primegaps.o fs64.o primegen.a math.lib
	./load primegaps fs64.o primegen.a  `cat math.lib`

primegaps.0: \
primegaps.1
	nroff -man primegaps.1 > primegaps.0

primegaps.o: \
compile primegaps.c primegen.h uint32.h uint64.h
	./compile primegaps.c

primegen.0: \
primegen.3
	nroff -man primegen.3 > primegen.0

primegen.a: \
makelib primegen.o primegen_init.o primegen_next.o primegen_skip.o
	./makelib primegen.a primegen.o primegen_init.o \
	primegen_next.o primegen_skip.o

primegen.h: \
conf-words primegen.h.sed
	sed s/conf-words/`head -1 conf-words`/ \
	< primegen.h.sed > primegen.h

primegen.o: \
compile primegen.c primegen.h uint32.h uint64.h primegen_impl.h \
int64.h
	./compile primegen.c

primegen_init.o: \
compile primegen_init.c primegen.h uint32.h uint64.h primegen_impl.h
	./compile primegen_init.c

primegen_next.o: \
compile primegen_next.c primegen.h uint32.h uint64.h primegen_impl.h
	./compile primegen_next.c

primegen_skip.o: \
compile primegen_skip.c primegen.h uint32.h uint64.h primegen_impl.h
	./compile primegen_skip.c

primes: \
load primes.o fs64.o primegen.a
	./load primes fs64.o primegen.a 

primes.0: \
primes.1
	nroff -man primes.1 > primes.0

primes.o: \
compile primes.c primegen.h uint32.h uint64.h fs64.h uint64.h
	./compile primes.c

primespeed: \
load primespeed.o fs64.o primegen.a
	./load primespeed fs64.o primegen.a 

primespeed.0: \
primespeed.1
	nroff -man primespeed.1 > primespeed.0

primespeed.o: \
compile primespeed.c timing.h hasrdtsc.h hasgethr.h primegen.h \
uint32.h uint64.h primegen_impl.h fs64.h uint64.h
	./compile primespeed.c

prog: \
primes primespeed primegaps eratspeed

setup: \
it install
	./install

shar: \
FILES BLURB README TODO THANKS CHANGES FILES TARGETS VERSION SYSDEPS \
Makefile INSTALL primes.1 primes.c primespeed.1 primespeed.c \
primegaps.1 primegaps.c primegen.3 conf-words primegen.h.sed \
primegen_impl.h primegen.c primegen_init.c primegen_next.c \
primegen_skip.c eratspeed.c fs64.h fs64.c math.lib hier.c conf-home \
auto-str.c auto_home.h install.c instcheck.c conf-cc conf-ld \
find-systype.sh make-compile.sh make-load.sh make-makelib.sh trycpp.c \
warn-auto.sh substdio.h substdio.c substdi.c substdo.c \
substdio_copy.c subfd.h subfderr.c readwrite.h exit.h strerr.h \
strerr_sys.c strerr_die.c error.3 error_str.3 error.h error.c \
error_str.c open.h open_read.c open_trunc.c byte.h byte_copy.c \
byte_cr.c str.h str_len.c uint32.h1 uint32.h2 tryulong32.c uint64.h1 \
uint64.h2 tryulong64.c int64.h1 int64.h2 trylong64.c timing.h \
tryrdtsc.c trygethr.c
	shar -m `cat FILES` > shar
	chmod 400 shar

str.a: \
makelib str_len.o byte_copy.o byte_cr.o
	./makelib str.a str_len.o byte_copy.o byte_cr.o

str_len.o: \
compile str_len.c str.h
	./compile str_len.c

strerr.a: \
makelib strerr_sys.o strerr_die.o
	./makelib strerr.a strerr_sys.o strerr_die.o

strerr_die.o: \
compile strerr_die.c substdio.h subfd.h substdio.h exit.h strerr.h
	./compile strerr_die.c

strerr_sys.o: \
compile strerr_sys.c error.h strerr.h
	./compile strerr_sys.c

subfderr.o: \
compile subfderr.c readwrite.h substdio.h subfd.h substdio.h
	./compile subfderr.c

substdi.o: \
compile substdi.c substdio.h byte.h error.h
	./compile substdi.c

substdio.a: \
makelib substdio.o substdi.o substdo.o subfderr.o substdio_copy.o
	./makelib substdio.a substdio.o substdi.o substdo.o \
	subfderr.o substdio_copy.o

substdio.o: \
compile substdio.c substdio.h
	./compile substdio.c

substdio_copy.o: \
compile substdio_copy.c substdio.h
	./compile substdio_copy.c

substdo.o: \
compile substdo.c substdio.h str.h byte.h error.h
	./compile substdo.c

systype: \
find-systype trycpp.c
	./find-systype > systype

uint32.h: \
tryulong32.c compile load uint32.h1 uint32.h2
	( ( ./compile tryulong32.c && ./load tryulong32 && \
	./tryulong32 ) >/dev/null 2>&1 \
	&& cat uint32.h2 || cat uint32.h1 ) > uint32.h
	rm -f tryulong32.o tryulong32

uint64.h: \
tryulong64.c compile load uint64.h1 uint64.h2
	( ( ./compile tryulong64.c && ./load tryulong64 && \
	./tryulong64 ) >/dev/null 2>&1 \
	&& cat uint64.h1 || cat uint64.h2 ) > uint64.h
	rm -f tryulong64.o tryulong64
