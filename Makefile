CC ?= cc
CFLAGS ?= -g -Wall -O2
CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -g
CARGO ?= cargo
RUSTFLAGS ?= -g
LDFLAGS = -lpthread

.PHONY: all clean

all: libmatrix.so libmandelbrot.so

clean:
	rm -f libmatrix.so libmandelbrot.so

libmatrix.so: matrix.cpp
	$(CXX) $(CXXFLAGS) -shared -fPIC -fopenmp -o $@ $^ $(LDFLAGS)

libmandelbrot.so: mandelbrot.cpp
	$(CXX) $(CXXFLAGS) -shared -fPIC -fopenmp -o $@ $^ $(LDFLAGS)

# Usually there is no need to modify this
check: all
	$(MAKE) -C tests check
