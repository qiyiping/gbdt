#include "gbdt.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include "time.hpp"
#include "cmd_option.hpp"
#include "loss.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace gbdt;

int main(int argc, char *argv[]) {
  CmdOption opt = CmdOption::ParseOptions(argc, argv);

#ifdef USE_OPENMP
  const int threads_wanted = opt.Get<int>("num_of_threads", 4);
  omp_set_num_threads(threads_wanted);
#endif
  std::srand ( unsigned ( std::time(0) ) );

  Configure conf;
  conf.number_of_feature = opt.Get<int>("feature_size", 0);
  conf.max_depth = opt.Get<int>("max_depth", 0);
  conf.iterations = opt.Get<int>("iterations", 0);
  conf.shrinkage = opt.Get<double>("shrinkage", 0.0);
  conf.feature_sample_ratio = opt.Get<double>("feature_ratio", 1.0);
  conf.data_sample_ratio = opt.Get<double>("data_ratio", 1.0);
  conf.debug = opt.Get<bool>("debug", false);
  conf.min_leaf_size = opt.Get<int>("min_leaf_size", 0);
  std::string loss_type = opt.Get<std::string>("loss", "");

  Objective *objective = LossFactory::GetInstance()->Create(loss_type);
  if (!objective) {
    LossFactory::GetInstance()->PrintAllCandidates();
    return -1;
  }

  conf.loss.reset(objective);

  std::cout << conf.ToString() << std::endl;

  std::string train_file = opt.Get<std::string>("train_file", "");
  if (train_file.empty()) {
    std::cerr << "please specify train file" << std::endl;
  }

  DataVector d;
  bool r = LoadDataFromFile(train_file,
                            &d,
                            conf.number_of_feature,
                            loss_type == std::string("LogLoss"));
  assert(r);

  GBDT gbdt(conf);

  Elapsed elapsed;
  gbdt.Fit(&d);
  std::cout << "fit time: " << elapsed.Tell().ToMilliseconds() << std::endl;
  CleanDataVector(&d);
  FreeVector(&d);

  std::string model_file = train_file + ".model";
  std::ofstream model_output(model_file.c_str());
  model_output << gbdt.Save();

  double *g = gbdt.GetGain();
  std::cout << "feature index\tfeature gain" << std::endl;
  for (size_t i = 0; i < conf.number_of_feature; ++i) {
    std::cout << i << "\t" << g[i] << std::endl;
  }

  return 0;
}
