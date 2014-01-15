#include "gbdt.hpp"
#include "fitness.hpp"
#include <iostream>
#include <cassert>
#include <boost/lexical_cast.hpp>
#include "time.hpp"

using namespace gbdt;

int main(int argc, char *argv[]) {
  gConf.number_of_feature = 3;
  gConf.max_depth = 4;
  gConf.iterations = 10;
  gConf.shrinkage = 0.1;

  if (argc > 1) {
    gConf.max_depth = boost::lexical_cast<int>(argv[1]);
  }

  if (argc > 2) {
    gConf.iterations = boost::lexical_cast<int>(argv[2]);
  }

  if (argc > 3) {
    gConf.shrinkage = boost::lexical_cast<float>(argv[3]);
  }

  DataVector d;
  bool r = LoadDataFromFile("../../data/test.dat", &d);
  assert(r);

  GBDT gbdt;

  Elapsed elapsed;
  gbdt.Fit(&d);
  std::cout << "fit time: " << elapsed.Tell() << std::endl;

  DataVector::iterator iter = d.begin();
  PredictVector predict;
  for ( ; iter != d.end(); ++iter) {
    ValueType p = gbdt.Predict(**iter);
    predict.push_back(p);
    std::cout << (*iter)->ToString() << std::endl
              << p << std::endl;
  }

  std::cout << "rmse: " << RMSE(d, predict) << std::endl;

  CleanDataVector(&d);
  return 0;
}
