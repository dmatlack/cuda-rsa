#####################################################################
#
# Multiple Precision Integers Library Makefile
#
#####################################################################

CXX = g++ -m64
CXXFLAGS = -g -DDEBUG -Wall -Wextra -I/usr/local/cuda/include -I../include

NVCC = nvcc
NVCCFLAGS = -g -G -m64 -arch compute_20

LIBS += cudart
LDLIBS = $(addprefix -l, $(LIBS))
LDFLAGS = -L/usr/local/cuda/lib64/ -lcudart

###
# Files we don't compile with directly, but we want Make to know that we
# depend on them
###
OTHER_FILES = cuda_string.h digit.h mpz.h

default: cpu

########################################################
#																											 #
# CPU test program																		 #
#																											 #
########################################################
cpu: $(OTHER_FILES) cpu_main.c
		mkdir -p obj
		$(CXX) $(CXXFLAGS) cpu_main.c -c -o obj/cpu_main.o
		$(CXX) $(CXXFLAGS) obj/cpu_main.o -o cpu
		./cpu

%.c:
%.h:
%.cu:
%.cpp:

clean:
	rm -rf obj/
	rm -f gpu
	rm -f cpu
