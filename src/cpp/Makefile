header_files = data.hpp fitness.hpp tree.hpp util.hpp config.hpp gbdt.hpp time.hpp auc.hpp
object_files = data.o fitness.o tree.o util.o config.o gbdt.o auc.o time.o

tests = data_unittest tree_unittest

CXX = g++
CXXFLAGS = -Wall -Wextra -Wconversion -pedantic -fopenmp -O2
#CXXFLAGS = -Wall -Wextra -Wconversion -pedantic -fopenmp -O2 -DUSE_OPENMP

all: libgbdt.a $(tests) ctr gbdt_predict gbdt_train

data.o: $(header_files) data.cpp
	$(CXX) -c $(CXXFLAGS) data.cpp

fitness.o: $(header_files) fitness.cpp
	$(CXX) -c $(CXXFLAGS) fitness.cpp

tree.o: $(header_files) tree.cpp
	$(CXX) -c $(CXXFLAGS) tree.cpp

util.o: $(header_files) util.cpp
	$(CXX) -c $(CXXFLAGS) util.cpp

config.o: $(header_files) config.cpp
	$(CXX) -c $(CXXFLAGS) config.cpp

gbdt.o: $(header_files) gbdt.cpp
	$(CXX) -c $(CXXFLAGS) gbdt.cpp

auc.o: $(header_files) auc.cpp
	$(CXX) -c $(CXXFLAGS) auc.cpp

time.o: $(header_files) time.cpp
	$(CXX) -c $(CXXFLAGS) time.cpp

libgbdt.a: $(object_files)
	ar rcs libgbdt.a $(object_files)

ctr: libgbdt.a ctr.cpp
	$(CXX) $(CXXFLAGS) -o ctr ctr.cpp libgbdt.a

gbdt_predict: libgbdt.a gbdt_predict.cpp
	$(CXX) $(CXXFLAGS) -o gbdt_predict gbdt_predict.cpp libgbdt.a


data_unittest: libgbdt.a data_unittest.cpp
	$(CXX) $(CXXFLAGS) -o data_unittest data_unittest.cpp libgbdt.a

tree_unittest: libgbdt.a tree_unittest.cpp
	$(CXX) $(CXXFLAGS) -o tree_unittest tree_unittest.cpp libgbdt.a

gbdt_train: libgbdt.a gbdt_train.cpp
	$(CXX) $(CXXFLAGS) -o gbdt_train gbdt_train.cpp libgbdt.a

clean:
	rm $(object_files) $(tests) libgbdt.a ctr gbdt_predict gbdt_train
