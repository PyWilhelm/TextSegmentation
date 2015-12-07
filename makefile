all:
	cython -3 -o testgit.c testgit.pyx
	gcc -g -O2 -fpic -c testgit.c -o testgit.o python3.4-config --cflags
	gcc -g -O2 -shared -o testgit.so testgit.o python3.4-config --libs