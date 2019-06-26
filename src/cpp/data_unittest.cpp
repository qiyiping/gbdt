#include "data.hpp"
#include <iostream>
#include <cassert>

using namespace gbdt;

int main(int argc, char *argv[]) {
  UNUSED(argc);
  UNUSED(argv);
  std::string l = "0 2 0:10 1:100";
  int number_of_feature = 3;
  bool two_class_classification = true;

  Tuple *t = Tuple::FromString(l,
                               number_of_feature,
                               two_class_classification);

  std::cout << t->ToString(number_of_feature) << std::endl;

  DataVector d;
  bool r = LoadDataFromFile("../../data/test.txt",
                            &d,
                            number_of_feature,
                            two_class_classification);
  assert(r);
  DataVector::iterator iter = d.begin();
  for ( ; iter != d.end(); ++iter) {
    std::cout << (*iter)->ToString(number_of_feature) << std::endl;
  }

  CleanDataVector(&d);
  return 0;
}
