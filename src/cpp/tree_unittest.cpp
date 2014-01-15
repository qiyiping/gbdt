#include "tree.hpp"
#include "fitness.hpp"
#include <iostream>
#include <cassert>
#include <boost/lexical_cast.hpp>

using namespace gbdt;

int main(int argc, char *argv[]) {
  gConf.number_of_feature = 3;
  gConf.max_depth = 4;
  if (argc > 1) {
    gConf.max_depth = boost::lexical_cast<int>(argv[1]);
  }

  DataVector d;
  bool r = LoadDataFromFile("../../data/test.dat", &d);
  assert(r);

  RegressionTree tree;

  tree.Fit(&d);

  DataVector::iterator iter = d.begin();
  PredictVector predict;
  for ( ; iter != d.end(); ++iter) {
    std::cout << (*iter)->ToString() << std::endl;
    ValueType p = tree.Predict(**iter);
    predict.push_back(p);
    std::cout << p << std::endl;
  }

  std::cout << "rmse: " << RMSE(d, predict) << std::endl;

  CleanDataVector(&d);
  return 0;
}
