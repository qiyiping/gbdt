#include "gbdt.hpp"
#include "fitness.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <boost/lexical_cast.hpp>
#include "time.hpp"

using namespace gbdt;

int main(int argc, char *argv[]) {
  std::srand ( unsigned ( std::time(0) ) );

  g_conf.number_of_feature = 3;
  g_conf.max_depth = 4;
  g_conf.iterations = 100;
  g_conf.shrinkage = 0.1F;

  if (argc < 3) return -1;

  std::string train_file(argv[1]);
  std::string test_file(argv[2]);

  if (argc > 3) {
    g_conf.max_depth = boost::lexical_cast<int>(argv[3]);
  }

  if (argc > 4) {
    g_conf.iterations = boost::lexical_cast<int>(argv[4]);
  }

  if (argc > 5) {
    g_conf.shrinkage = boost::lexical_cast<float>(argv[5]);
  }

  if (argc > 6) {
    g_conf.feature_sample_ratio = boost::lexical_cast<float>(argv[6]);
  }

  if (argc > 7) {
    g_conf.data_sample_ratio = boost::lexical_cast<float>(argv[7]);
  }

  g_conf.debug = true;
  // g_conf.loss = LOG_LIKELIHOOD;
  g_conf.loss = SQUARED_ERROR;

  DataVector d;
  bool r = LoadDataFromFile(train_file, &d);
  assert(r);

  // g_conf.min_leaf_size = d.size() / 10;

  std::cout << g_conf.ToString() << std::endl;

  GBDT gbdt;

  Elapsed elapsed;
  gbdt.Fit(&d);
  std::cout << "fit time: " << elapsed.Tell() << std::endl;
  CleanDataVector(&d);
  FreeVector(&d);

  std::string model_file = train_file + ".model";
  std::ofstream model_output(model_file.c_str());
  model_output << gbdt.Save();

  double *g = gbdt.GetGain();
  for (size_t i = 0; i < g_conf.number_of_feature; ++i) {
    std::cout << i << "\t" << g[i] << std::endl;
  }

  GBDT gbdt2;
  gbdt2.Load(gbdt.Save());

  DataVector d2;
  r = LoadDataFromFile(test_file, &d2);
  assert(r);

  elapsed.Reset();
  DataVector::iterator iter = d2.begin();
  PredictVector predict;
  for ( ; iter != d2.end(); ++iter) {
    ValueType p;
    if (g_conf.loss == SQUARED_ERROR) {
      p = gbdt2.Predict(**iter);
      predict.push_back(p);
    } else if (g_conf.loss == LOG_LIKELIHOOD) {
      p = gbdt2.Predict(**iter);
      p = Logit(p);
      if (p >= 0.5)
        p = 1;
      else
        p = -1;
      predict.push_back(p);
    }
    // std::cout << (*iter)->ToString() << std::endl
    //           << p << std::endl;
  }

  std::cout << "predict time: " << elapsed.Tell() << std::endl;
  std::cout << "rmse: " << RMSE(d2, predict) << std::endl;

  CleanDataVector(&d2);

  return 0;
}
