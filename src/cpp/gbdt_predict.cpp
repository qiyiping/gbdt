// Author: qiyiping@gmail.com (Yiping Qi)

#include "gbdt.hpp"
#include "fitness.hpp"
#include <fstream>
#include <cassert>
#include <cstring>
#include <iostream>
#include <algorithm>

#include "auc.hpp"

#include "cmd_option.hpp"

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

  GBDT gbdt;
  gbdt.Load(model);

  size_t feature_num = opt.Get<size_t>("feature_size", 0);

  g_conf.number_of_feature = feature_num;

  DataVector d;
  std::string input_file = opt.Get<std::string>("input", "");
  LoadDataFromFile(input_file, &d);

  std::string loss_str = opt.Get<std::string>("loss", "");
  g_conf.loss = StringToLoss(loss_str);
  if (g_conf.loss == UNKNOWN_LOSS) {
    std::cerr << "unknown loss type: " << loss_str << std::endl
              << "possible loss type are SQUARED_ERROR, LOG_LIKELIHOOD and LAD" << std::endl;
    return -1;
  }

  int debug = opt.Get<int>("debug", 0);

  std::cout << g_conf.ToString() << std::endl;

  Loss loss_type = g_conf.loss;

  DataVector::iterator iter = d.begin();

  std::string predict_file = input_file + ".predict";
  std::ofstream predict_output(predict_file.c_str());

  Auc auc;
  double sum = 0.0;
  double cnt = 0.0;
  double *gain = new double[feature_num];

  for ( ; iter != d.end(); ++iter) {
    ValueType p = 0;

    if (debug) {
      std::fill_n(gain, feature_num, 0.0);
      p = gbdt.Predict(**iter, gain);
    } else {
      p = gbdt.Predict(**iter);
    }

    if (loss_type == SQUARED_ERROR) {
      sum += Squared(p - (*iter)->label) * (*iter)->weight;
      cnt += (*iter)->weight;
    } else if (loss_type == LOG_LIKELIHOOD) {
      p = Logit(p);
      auc.Add(p, (*iter)->label);
    } else if (loss_type == LAD) {
      sum += Abs(p - (*iter)->label) * (*iter)->weight;
      cnt += (*iter)->weight;
    }

    predict_output << "--------------------------" << std::endl
                   << p << std::endl;
    if (debug) {
      for (int i = 0; i < feature_num; ++i) {
        predict_output << i << ":" << gain[i] << " ";
      }
      predict_output << std::endl;
    }
    predict_output <<(*iter)->ToString() << std::endl;
  }

  delete[] gain;

  if (loss_type == SQUARED_ERROR) {
    std::cout << "rmse: " << std::sqrt(sum / cnt) << std::endl;
  } else if (loss_type == LOG_LIKELIHOOD) {
    std::cout << "auc: " << auc.CalculateAuc() << std::endl;
    auc.PrintConfusionTable();
  } else if (loss_type == LAD) {
    std::cout << "mae: " << sum / cnt << std::endl;
  }

  CleanDataVector(&d);
  FreeVector(&d);
  return 0;
}
