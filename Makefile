###############################################################################
#
# 15-418 Final Project Makefile
#
# AJ Kaufmann
# David Matlack
#
###############################################################################

SRC_DIR = src
OBJ_DIR = obj

CXX = g++ -m64

CXXFLAGS = -g -DDEBUG -Wall -Wextra 						\
	-I/usr/local/cuda/include 										\
	-I/usr/local/cuda/CUDALibraries/common/inc 		\
	-I$(SRC_DIR)/primegen-0.97 										\
	-I$(SRC_DIR)/mpz/

NVCC = nvcc
NVCCFLAGS = -g -G -m64 -arch compute_20 				\
	-I$(SRC_DIR)/mpz

LDFLAGS = -L/usr/local/cuda/lib64/ -lcudart -L../primegen-0.97/lib -lprimegen

default: mpz

#################################################
# MPZ Library																		#
#################################################

MPZ_SRC = \
	mpz/compile.h 			\
	mpz/cuda_string.h 	\
	mpz/digit.h 				\
	mpz/mpz.h

mpz:
		mkdir -p obj
		$(CXX) $(CXXFLAGS) $(SRC_DIR)/mpz/cpu_main.c -c -o $(OBJ_DIR)/cpu_main.o
		$(CXX) $(CXXFLAGS) $(OBJ_DIR)/cpu_main.o -o mpz_test
		./mpz_test

clean:
	rm -f mpz_test
	rm -rf $(OBJ_DIR)
