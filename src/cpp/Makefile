header_files = data.hpp math_util.hpp tree.hpp util.hpp config.hpp gbdt.hpp time.hpp auc.hpp loss.hpp
object_files = data.o math_util.o tree.o util.o config.o gbdt.o auc.o time.o loss.o common_loss.o metrics.o

tests = data_unittest tree_unittest loss_unittest
execs = gbdt_predict gbdt_train

CXX = g++-5
# CXXFLAGS = -Wall -Wextra -Wconversion -pedantic -fopenmp -O2 -std=c++11
CXXFLAGS = -Wall -Wextra -Wconversion -pedantic -fopenmp -O2 -DUSE_OPENMP -std=c++11
# CXXFLAGS = -Wall -fopenmp -DUSE_OPENMP -std=c++11 -g -O0

all: libgbdt.a $(tests) $(execs) libcustom_loss_example.so

data.o: $(header_files) data.cpp
	$(CXX) -c $(CXXFLAGS) data.cpp

math_util.o: $(header_files) math_util.cpp
	$(CXX) -c $(CXXFLAGS) math_util.cpp

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

loss.o: $(header_files) loss.cpp
	$(CXX) -c $(CXXFLAGS) loss.cpp

metrics.o: $(header_files) metrics.cpp
	$(CXX) -c $(CXXFLAGS) metrics.cpp

common_loss.o: $(header_files) common_loss.cpp
	$(CXX) -c $(CXXFLAGS) common_loss.cpp

libgbdt.a: $(object_files)
	ar rcs libgbdt.a $(object_files)

data_unittest: libgbdt.a data_unittest.cpp
	$(CXX) $(CXXFLAGS) -o data_unittest data_unittest.cpp libgbdt.a

tree_unittest: libgbdt.a tree_unittest.cpp
	$(CXX) $(CXXFLAGS) -o tree_unittest tree_unittest.cpp libgbdt.a common_loss.o

loss_unittest: libgbdt.a loss_unittest.cpp
	$(CXX) $(CXXFLAGS) -o loss_unittest loss_unittest.cpp libgbdt.a common_loss.o

gbdt_train: libgbdt.a gbdt_train.cpp cmd_option.hpp
	$(CXX) $(CXXFLAGS) -o gbdt_train gbdt_train.cpp libgbdt.a common_loss.o

gbdt_predict: libgbdt.a gbdt_predict.cpp cmd_option.hpp
	$(CXX) $(CXXFLAGS) -o gbdt_predict gbdt_predict.cpp libgbdt.a common_loss.o

libcustom_loss_example.so: custom_loss_example.hpp loss.hpp custom_loss_example.cpp
	$(CXX) $(CXXFLAGS) -fPIC -shared custom_loss_example.cpp libgbdt.a -o libcustom_loss_example.so

clean:
	rm $(object_files) $(tests) libgbdt.a gbdt_predict gbdt_train
