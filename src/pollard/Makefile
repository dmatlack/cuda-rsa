#####################################################################
#
# Multiple Precision Integers Library Makefile
#
#####################################################################

CXX = g++ -m64
CXXFLAGS = -g -DDEBUG -Wall -Wextra \
	-I/usr/local/cuda/include \
	-I../mpz \
	-I/usr/local/cuda/CUDALibraries/common/inc

NVCC = nvcc
NVCCFLAGS = -g -G -m64 -arch compute_20 \
	-I../primegen-0.97 \
	-I../mpz

LIBS += cudart primegen
LDLIBS = $(addprefix -l, $(LIBS))
LDFLAGS = -L/usr/local/cuda/lib64/ -L../primegen-0.97/lib

###
# Files we don't compile with directly, but we want Make to know that we
# depend on them
###
OTHER_FILES = kernel.h ../mpz/*.h

default: dirs gpu

########################################################
#																											 #
# GPU test program																		 #
#																											 #
########################################################
gpu: $(OTHER_FILES) kernel.cu gpu_main.c
		$(NVCC) $(NVCCFLAGS) kernel.cu -c -o obj/kernel.o
		$(CXX) $(CXXFLAGS) $(LDFLAGS) gpu_main.c -c -o obj/gpu_main.o $(LDLIBS)
		$(CXX) $(LDFLAGS) obj/gpu_main.o obj/kernel.o -o gpu $(LDLIBS)


########################################################
#																											 #
# CPU test program																		 #
#																											 #
########################################################
dirs:
	mkdir -p obj

%.c:
%.h:
%.cu:
%.cpp:

clean:
	rm -rf obj/
	rm -f gpu
	rm -f cpu
