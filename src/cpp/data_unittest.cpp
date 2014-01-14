#include "data.hpp"
#include <iostream>
using namespace gbdt;

int main(int argc, char *argv[]) {
  std::string l = "1 2 2:10 3:100";
  gConf.number_of_feature = 10;
  gConf.max_depth = 4;

  Tuple *t = Tuple::FromString(l);

  std::cout << t->ToString() << std::endl;
  return 0;
}
