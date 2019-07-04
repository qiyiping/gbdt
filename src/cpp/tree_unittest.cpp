#include "tree.hpp"
#include <iostream>
#include <cassert>
#include <fstream>

#include "loss.hpp"
#include "common_loss.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace gbdt;

int main(int argc, char *argv[]) {
#ifdef USE_OPENMP
  const int threads_wanted = 4;
  omp_set_num_threads(threads_wanted);
#endif

  Configure conf;
  conf.number_of_feature = 3;
  conf.max_depth = 4;
  conf.loss.reset(LossFactory::GetInstance()->Create("SquaredError"));

  std::cout << conf.ToString() << std::endl;

  DataVector d;
  bool r = LoadDataFromFile("../../data/train.txt",
                            &d,
                            conf.number_of_feature,
                            false);
  assert(r);
  // setup target
  DataVector::iterator iter = d.begin();
  for ( ; iter != d.end(); ++iter) {
    (*iter)->target = (*iter)->label;
  }

  RegressionTree tree(conf);

  tree.Fit(&d);
  std::ofstream model_output("../../data/model");
  model_output << tree.Save();

  RegressionTree tree2(conf);
  tree2.Load(tree.Save());

  DataVector d2;
  r = LoadDataFromFile("../../data/test.txt",
                       &d2,
                       conf.number_of_feature,
                       false);
  assert(r);

  iter = d2.begin();
  PredictVector predict;
  for ( ; iter != d2.end(); ++iter) {
    // std::cout << (*iter)->ToString(conf.number_of_feature) << std::endl;
    ValueType p = tree2.Predict(**iter);
    predict.push_back(p);
    // std::cout << p << "," << tree.Predict(**iter) << std::endl;
  }

  std::cout << "rmse: " << RMSE(d2, predict) << std::endl;

  CleanDataVector(&d);
  CleanDataVector(&d2);
  return 0;
}
