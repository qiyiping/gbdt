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
#include <cstring>

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace gbdt;

int main(int argc, char *argv[]) {
  std::srand ( unsigned ( std::time(0) ) );

#ifdef USE_OPENMP
  const int threads_wanted = 4;
  omp_set_num_threads(threads_wanted);
#endif

  g_conf.number_of_feature = 62;
  g_conf.max_depth = 5;
  g_conf.iterations = 300;
  g_conf.shrinkage = 0.1F;

  if (argc < 3) return -1;

  std::string train_file(argv[1]);
  std::string test_file(argv[2]);

  std::cout << "training file: " << train_file << std::endl;
  std::cout << "testing file: " << test_file << std::endl;

  if (argc > 3) {
    g_conf.max_depth = boost::lexical_cast<size_t>(argv[3]);
  }

  if (argc > 4) {
    g_conf.iterations = boost::lexical_cast<size_t>(argv[4]);
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

  int debug = 0;
  if (argc > 8) {
    debug = boost::lexical_cast<int>(argv[8]);
  }
  g_conf.debug = debug > 0? true : false;

  if (argc > 9) {
    if (std::strncmp("loglikelihood", argv[9], 20) == 0) {
      g_conf.loss = LOG_LIKELIHOOD;
    } else if (std::strncmp("squared", argv[9], 20) == 0) {
      g_conf.loss = SQUARED_ERROR;
    } else {
      std::cerr << "Unknown loss type" << std::endl;
      return -1;
    }
  }

  if (argc > 10) {
    g_conf.number_of_feature = boost::lexical_cast<size_t>(argv[10]);
  }

  int max_leafs = 40;
  if (argc > 11) {
    max_leafs = boost::lexical_cast<int>(argv[11]);
  }

  int enable_initial_guess = 0;
  if (argc > 12) {
    enable_initial_guess = boost::lexical_cast<int>(argv[12]);
  }
  g_conf.enable_initial_guess = enable_initial_guess > 0? true : false;

  if (argc > 13) {
    g_conf.LoadFeatureCost(argv[13]);
  }

  DataVector d;
  bool r = LoadDataFromFile(train_file, &d, true);
  assert(r);

  g_conf.min_leaf_size = d.size() * g_conf.data_sample_ratio / max_leafs;
  std::cout << "configure: " << std::endl
            << g_conf.ToString() << std::endl;

  GBDT gbdt;
  Elapsed elapsed;
  gbdt.Fit(&d);
  std::cout << "fit time: " << elapsed.Tell().ToMilliseconds() << std::endl;

  std::string model_file = train_file + ".model";
  std::ofstream model_output(model_file.c_str());
  model_output << gbdt.Save();

  CleanDataVector(&d);
  FreeVector(&d);

  double *g = gbdt.GetGain();
  for (size_t i = 0; i < g_conf.number_of_feature; ++i) {
    std::cout << i << "\t" << g[i] << std::endl;
  }

  DataVector d2;
  r = LoadDataFromFile(test_file, &d2, true);
  assert(r);

  elapsed.Reset();
  DataVector::iterator iter = d2.begin();
  PredictVector predict;
  for ( ; iter != d2.end(); ++iter) {
    ValueType t = gbdt.Predict(**iter);
    if (g_conf.loss == LOG_LIKELIHOOD) {
      predict.push_back(Logit(t));
    } else {
      predict.push_back(t);
    }
  }
  std::cout << "predict time: " << elapsed.Tell().ToMilliseconds() << std::endl;

  std::string predict_file = test_file + ".predict";
  std::ofstream predict_output(predict_file.c_str());

  Auc auc;
  for (size_t i = 0; i < d2.size(); ++i) {
    predict_output << predict[i] << " " << d2[i]->ToString() << std::endl;
    for (int s = 0; s < d2[i]->weight; ++s) {
      auc.Add(predict[i], d2[i]->label);
    }
  }
  std::cout << "auc: " << auc.CalculateAuc() << std::endl;
  auc.PrintConfusionTable();

  CleanDataVector(&d2);

  return 0;
}
