// Author: qiyiping@gmail.com (Yiping Qi)

#include "gbdt.hpp"
#include "math_util.hpp"
#include <fstream>
#include <cassert>
#include <cstring>
#include <iostream>
#include <algorithm>

#include "auc.hpp"
#include "cmd_option.hpp"
#include "loss.hpp"

using namespace gbdt;

int main(int argc, char *argv[]) {
  CmdOption opt = CmdOption::ParseOptions(argc, argv);

  std::string model;
  std::ifstream stream(opt.Get<std::string>("model", ""));
  assert(stream);

  stream.seekg(0, std::ios::end);
  model.reserve(stream.tellg());
  stream.seekg(0, std::ios::beg);
  model.assign(std::istreambuf_iterator<char>(stream),
               std::istreambuf_iterator<char>());

  Configure conf;
  conf.number_of_feature = opt.Get<size_t>("feature_size", 0);
  std::string loss_type = opt.Get<std::string>("loss", "");
  Objective *objective = LossFactory::GetInstance()->Create(loss_type);
  if (!objective) {
    LossFactory::GetInstance()->PrintAllCandidates();
    return -1;
  }

  conf.loss.reset(objective);

  std::cout << conf.ToString() << std::endl;

  GBDT gbdt(conf);
  gbdt.Load(model);

  DataVector d;
  std::string input_file = opt.Get<std::string>("input", "");
  LoadDataFromFile(input_file,
                   &d,
                   conf.number_of_feature,
                   loss_type == std::string("LogLoss"));

  int debug = opt.Get<int>("debug", 0);

  DataVector::iterator iter = d.begin();

  std::string predict_file = input_file + ".predict";
  std::ofstream predict_output(predict_file.c_str());

  Auc auc;
  double sum = 0.0;
  double cnt = 0.0;
  double *gain = new double[conf.number_of_feature];

  for ( ; iter != d.end(); ++iter) {
    ValueType p = 0;

    if (debug) {
      std::fill_n(gain, conf.number_of_feature, 0.0);
      p = gbdt.Predict(**iter, gain);
    } else {
      p = gbdt.Predict(**iter);
    }

    if (loss_type == std::string("SquaredError")) {
      sum += Squared(p - (*iter)->label) * (*iter)->weight;
      cnt += (*iter)->weight;
    } else if (loss_type == std::string("LogLoss")) {
      p = Logit(p);
      auc.Add(p, (*iter)->label);
    } else if (loss_type == std::string("LAD")) {
      sum += Abs(p - (*iter)->label) * (*iter)->weight;
      cnt += (*iter)->weight;
    }

    predict_output << "--------------------------" << std::endl
                   << p << std::endl;
    if (debug) {
      for (int i = 0; i < conf.number_of_feature; ++i) {
        predict_output << i << ":" << gain[i] << " ";
      }
      predict_output << std::endl;
    }
    predict_output <<(*iter)->ToString(conf.number_of_feature) << std::endl;
  }

  delete[] gain;

  if (loss_type == std::string("SquaredError")) {
    std::cout << "rmse: " << std::sqrt(sum / cnt) << std::endl;
  } else if (loss_type == std::string("LogLoss")) {
    std::cout << "auc: " << auc.CalculateAuc() << std::endl;
    auc.PrintConfusionTable();
  } else if (loss_type == std::string("LAD")) {
    std::cout << "mae: " << sum / cnt << std::endl;
  }

  CleanDataVector(&d);
  FreeVector(&d);
  return 0;
}
