// Author: qiyiping@gmail.com (Yiping Qi)
#include "gbdt.hpp"
#include "fitness.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <boost/lexical_cast.hpp>
#include "time.hpp"
#include "auc.hpp"

#include <cstdlib>
#include <ctime>

using namespace gbdt;

int main(int argc, char *argv[]) {
  std::srand ( unsigned ( std::time(0) ) );

  gConf.number_of_feature = 66;
  gConf.max_depth = 4;
  gConf.iterations = 10;
  gConf.shrinkage = 0.1;

  if (argc < 3) return -1;

  std::string train_file(argv[1]);
  std::string test_file(argv[2]);

  if (argc > 3) {
    gConf.max_depth = boost::lexical_cast<int>(argv[3]);
  }

  if (argc > 4) {
    gConf.iterations = boost::lexical_cast<int>(argv[4]);
  }

  if (argc > 5) {
    gConf.shrinkage = boost::lexical_cast<float>(argv[5]);
  }

  if (argc > 6) {
    gConf.feature_sample_ratio = boost::lexical_cast<float>(argv[6]);
  }

  DataVector d;
  bool r = LoadDataFromFile(train_file, &d);
  assert(r);

  GBDT gbdt;
  Elapsed elapsed;
  gbdt.Fit(&d);
  std::cout << "fit time: " << elapsed.Tell() << std::endl;

  std::string model_file = train_file + ".model";
  std::ofstream model_output(model_file.c_str());
  model_output << gbdt.Save();

  DataVector d2;
  r = LoadDataFromFile(test_file, &d2);
  assert(r);

  elapsed.Reset();
  DataVector::iterator iter = d2.begin();
  PredictVector predict;
  Auc auc;
  for ( ; iter != d2.end(); ++iter) {
    ValueType p = gbdt.Predict(**iter);
    predict.push_back(p);
    auc.Add(p, (*iter)->label);
  }
  std::cout << "predict time: " << elapsed.Tell() << std::endl;

  std::cout << "auc: " << auc.CalculateAuc() << std::endl;

  CleanDataVector(&d2);
  CleanDataVector(&d);

  return 0;
}
