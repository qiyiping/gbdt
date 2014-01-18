#include "data.hpp"
#include <iostream>
#include <cassert>

using namespace gbdt;

int main(int argc, char *argv[]) {
  UNUSED(argc);
  UNUSED(argv);
  std::string l = "0 2 0:10 1:100";
  g_conf.number_of_feature = 3;
  g_conf.max_depth = 4;
  g_conf.loss = LOG_LIKELIHOOD;

  Tuple *t = Tuple::FromString(l);

  std::cout << t->ToString() << std::endl;

  DataVector d;
  bool r = LoadDataFromFile("../../data/test.dat", &d);
  assert(r);
  DataVector::iterator iter = d.begin();
  for ( ; iter != d.end(); ++iter) {
    std::cout << (*iter)->ToString() << std::endl;
  }

  CleanDataVector(&d);
  return 0;
}
