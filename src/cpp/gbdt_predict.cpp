// Author: qiyiping@gmail.com (Yiping Qi)

#include "gbdt.hpp"
#include <fstream>
#include <cassert>
#include <cstring>
#include <iostream>
#include <algorithm>

#include "cmd_option.hpp"
#include "loss.hpp"
#include "common_loss.hpp"
#include "metrics.hpp"

using namespace gbdt;

int main(int argc, char *argv[]) {
  CmdOption opt;
  opt.AddOption("model", "m", "model", "");
  opt.AddOption("feature_size", "s", "feature_size", OptionType::INT, true);
  opt.AddOption("input", "i", "input", "");
  opt.AddOption("metric", "t", "metric", "");
  opt.AddOption("classification", "c", "classification", false);

  if (!opt.ParseOptions(argc, argv)) {
    opt.Help();
    return -1;
  }

  Configure conf;
  opt.Get("feature_size", &conf.number_of_feature);
  std::cout << conf.ToString() << std::endl;
  bool classification;
  opt.Get("classification", &classification);
  GBDT gbdt(conf);

  std::string model_path;
  opt.Get("model", &model_path);
  std::string model;
  std::ifstream stream(model_path);
  assert(stream);

  stream.seekg(0, std::ios::end);
  model.reserve(stream.tellg());
  stream.seekg(0, std::ios::beg);
  model.assign(std::istreambuf_iterator<char>(stream),
               std::istreambuf_iterator<char>());

  gbdt.Load(model);

  DataVector d;
  std::string input_file;
  opt.Get("input", &input_file);
  LoadDataFromFile(input_file,
                   &d,
                   conf.number_of_feature,
                   classification);

  std::string metric;
  opt.Get("metric", &metric);

  std::string predict_file = input_file + ".predict";
  std::ofstream predict_output(predict_file.c_str());
  PredictVector pv;

  for (auto iter = d.begin() ; iter != d.end(); ++iter) {
    ValueType p = gbdt.Predict(**iter);

    predict_output << "--------------------------" << std::endl
                   << p << " " << (*iter)->ToString(conf.number_of_feature) << std::endl;
    pv.push_back(p);
  }

  if (metric == "auc") {
    std::cout << metric << ": " << Metrics::AucScore(d, pv, d.size()) << std::endl;
  } else if (metric == "logloss") {
    std::cout << metric << ": " << Metrics::LogLoss(d, pv, d.size()) << std::endl;
  } else if (metric == "mse") {
    std::cout << metric << ": " << Metrics::MeanSquaredError(d, pv, d.size()) << std::endl;
  } else if (metric == "mae") {
    std::cout << metric << ": " << Metrics::MeanAbsoluteError(d, pv, d.size()) << std::endl;
  }

  CleanDataVector(&d);
  FreeVector(&d);
  return 0;
}
