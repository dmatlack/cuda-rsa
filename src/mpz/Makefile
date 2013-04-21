#####################################################################
#
# Multiple Precision Integers Library Makefile
#
# by David Matlack
#
#####################################################################

EXE = main

SRC = digit.c mpz.c main.c
OBJ = $(SRC:.c=.o)

SRC_DIR = .
OBJ_DIR = obj

SRC_FULL = $(addprefix $(SRC_DIR)/, $(SRC))
OBJ_FULL = $(addprefix $(OBJ_DIR)/, $(OBJ))

CC = gcc
CCFLAGS += -Wall -Wextra -Werror -g -DDEBUG
LDFLAGS += 

.PHONY: target

$(shell mkdir -p $(OBJ_DIR))

target: $(EXE)

$(EXE): $(OBJ_FULL)
	$(CC) $(CCFLAGS) $(LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CCFLAGS) -c -o $@ $<

clean:
	rm -rf $(EXE) $(OBJ_DIR)
