#include "tree.hpp"
#include <iostream>
#include <cassert>
using namespace gbdt;

int main(int argc, char *argv[]) {
  gConf.number_of_feature = 3;
  gConf.max_depth = 2;

  DataVector d;
  bool r = LoadDataFromFile("../../data/test.dat", &d);
  assert(r);

  Node root;

  Node::Fit(&d, &root, 1);

  DataVector::iterator iter = d.begin();
  for ( ; iter != d.end(); ++iter) {
    std::cout << (*iter)->ToString() << std::endl;
    std::cout << Node::Predict(&root, **iter) << std::endl;
  }

  CleanDataVector(&d);
  return 0;
}
