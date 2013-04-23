CXX		=g++
LIBS		=-lgmp
CFLAGS		=-O4 -Wall -I.

OBJS=test1.o Integer.o

default:
	make test1

test1: $(OBJS)
	$(CXX) $(CFLAGS) $(DEFINE) -o test1 $(OBJS) $(LIBS)

clean:
	rm -f *.o test1 $(OBJS)
