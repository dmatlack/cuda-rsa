g++ -m64 -g -DDEBUG -Wall -Wextra pollard_p1.cc -c -o pollard_p1.o; g++ -m64 -g -DDEBUG -Wall -Wextra pollard_p1.o Integer.o -o pollard_p1 -lgmp; ./pollard_p1 1284717778837 >> out.txt
