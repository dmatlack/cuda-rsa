#####################################################################
#
# Multiple Precision Integers Library Makefile
#
#####################################################################

CXX=g++ -m64
CXXFLAGS=-g -DDEBUG -Wall -Wextra -I/usr/local/cuda/include

NVCC = nvcc
NVCCFLAGS = -g -G -m64 -arch compute_20

LIBS += cudart
LDLIBS = $(addprefix -l, $(LIBS))
LDFLAGS = -L/usr/local/cuda/lib64/ -lcudart

###
# Files we don't compile with directly, but we want Make to know that we
# depend on them
###
CUDA_FILES = cuda_string.h digit.h mpz.h

###
# Where our kernel is written
###
KERNEL = kernel.cu

###
# The C entrypoint of the program
###
MAIN = main.cpp

default: $(CUDA_FILES) $(KERNEL) $(MAIN)
		mkdir -p obj
		$(NVCC) $(NVCCFLAGS) $(KERNEL) -c -o obj/$(KERNEL:.cu=.o)
		$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LDLIBGS) $(MAIN) -c -o obj/$(MAIN:.cpp=.o)
		$(CXX) $(LDFLAGS) $(LDLIBGS) obj/* -o main

%.c:
%.h:
%.cu:
%.cpp:

clean:
	rm -rf obj/
	rm -f main
