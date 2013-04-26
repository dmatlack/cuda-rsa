#####################################################################
#
# Multiple Precision Integers Library Makefile
#
#####################################################################

CXX = g++ -m64
CXXFLAGS = -g -DDEBUG -Wall -Wextra -I/usr/local/cuda/include -lgmp

NVCC = nvcc
NVCCFLAGS = -g -G -m64 -arch compute_20

LIBS += cudart
LDLIBS = $(addprefix -l, $(LIBS))
LDFLAGS = -L/usr/local/cuda/lib64/ -lcudart

###
# Files we don't compile with directly, but we want Make to know that we
# depend on them
###
OTHER_FILES =

default: dirs gpu cpu

########################################################
#																											 #
# GPU test program																		 #
#																											 #
########################################################
gpu: $(OTHER_FILES) kernel.cu gpu_main.c
		$(NVCC) $(NVCCFLAGS) kernel.cu -c -o obj/kernel.o
		$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LDLIBGS) gpu_main.c -c -o obj/gpu_main.o
		$(CXX) $(LDFLAGS) $(LDLIBS) obj/gpu_main.o obj/kernel.o -o gpu
		./gpu

########################################################
#																											 #
# CPU test program																		 #
#																											 #
########################################################
cpu: $(OTHER_FILES) pollard.cpp
		$(CXX) $(CXXFLAGS) pollard.cpp -c -o obj/pollard.o
		$(CXX) $(CXXFLAGS) obj/pollard.o -o cpu

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
