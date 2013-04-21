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

CUDA_FILES = digit.cu string.cu mpz.cu
KERNEL = kernel.cu
MAIN = main.cpp

default: $(CUDA_FILES) $(KERNEL) $(MAIN)
		mkdir -p obj
		$(NVCC) $(NVCCFLAGS) $(KERNEL) -c -o obj/$(KERNEL:.cu=.o)
		$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LDLIBGS) $(MAIN) -c -o obj/$(MAIN:.cpp=.o)
		$(CXX) $(LDFLAGS) $(LDLIBGS) obj/* -o main

%.cu:
%.cpp:

clean:
	rm -rf obj/
	rm -f main
