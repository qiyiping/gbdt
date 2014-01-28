#include "tree.hpp"
#include "fitness.hpp"
#include <iostream>
#include <cassert>
#include <boost/lexical_cast.hpp>
#include <fstream>

using namespace gbdt;

int main(int argc, char *argv[]) {
  g_conf.number_of_feature = 3;
  g_conf.max_depth = 4;
  if (argc > 1) {
    g_conf.max_depth = boost::lexical_cast<int>(argv[1]);
  }

  std::cout << g_conf.ToString() << std::endl;

  DataVector d;
  bool r = LoadDataFromFile("../../data/train.dat", &d);
  assert(r);
  // setup target
  DataVector::iterator iter = d.begin();
  for ( ; iter != d.end(); ++iter) {
    (*iter)->target = (*iter)->label;
  }

  RegressionTree tree;

  tree.Fit(&d);
  std::ofstream model_output("../../data/model");
  model_output << tree.Save();

  RegressionTree tree2;
  tree2.Load(tree.Save());

  DataVector d2;
  r = LoadDataFromFile("../../data/test.dat", &d2);
  assert(r);

  iter = d2.begin();
  PredictVector predict;
  for ( ; iter != d2.end(); ++iter) {
    std::cout << (*iter)->ToString() << std::endl;
    ValueType p = tree2.Predict(**iter);
    predict.push_back(p);
    // std::cout << p << "," << tree.Predict(**iter) << std::endl;
  }

  std::cout << "rmse: " << RMSE(d2, predict) << std::endl;

  CleanDataVector(&d);
  CleanDataVector(&d2);
  return 0;
}
