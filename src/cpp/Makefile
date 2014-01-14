header_files = data.hpp fitness.hpp tree.hpp util.hpp
object_files = data.o fitness.o tree.o util.o

tests = data_unittest tree_unittest

CXX = g++
#CXXFLAGS = -Wall -Wextra -Wconversion -Werror -pedantic -O0 -g
CXXFLAGS = -Wall -Wextra -Wconversion -pedantic -O0 -g

all: libgbdt.a $(tests)

data.o: $(header_files) data.cpp
	$(CXX) -c $(CXXFLAGS) data.cpp

fitness.o: $(header_files) fitness.cpp
	$(CXX) -c $(CXXFLAGS) fitness.cpp

tree.o: $(header_files) tree.cpp
	$(CXX) -c $(CXXFLAGS) tree.cpp

util.o: $(header_files) util.cpp
	$(CXX) -c $(CXXFLAGS) util.cpp

libgbdt.a: $(object_files)
	ar rcs libgbdt.a $(object_files)




data_unittest: libgbdt.a data_unittest.cpp
	$(CXX) $(CXXFLAGS) -o data_unittest data_unittest.cpp libgbdt.a

tree_unittest: libgbdt.a tree_unittest.cpp
	$(CXX) $(CXXFLAGS) -o tree_unittest tree_unittest.cpp libgbdt.a

clean:
	rm $(object_files) $(tests) libgbdt.a
