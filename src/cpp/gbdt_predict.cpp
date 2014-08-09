// Author: qiyiping@gmail.com (Yiping Qi)

#include "gbdt.hpp"
#include "fitness.hpp"
#include <fstream>
#include <cassert>
#include <cstring>
#include <iostream>
#include <boost/lexical_cast.hpp>

#include "auc.hpp"

using namespace gbdt;

int main(int argc, char *argv[]) {
  std::string model;
  std::ifstream stream(argv[1]);
  assert(stream);

  stream.seekg(0, std::ios::end);
  model.reserve(stream.tellg());
  stream.seekg(0, std::ios::beg);
  model.assign(std::istreambuf_iterator<char>(stream),
               std::istreambuf_iterator<char>());

  GBDT gbdt;
  gbdt.Load(model);

  size_t feature_num = boost::lexical_cast<size_t>(argv[2]);

  g_conf.number_of_feature = feature_num;

  DataVector d;
  std::string input_file = argv[3];
  LoadDataFromFile(input_file, &d);

  Loss loss_type = SQUARED_ERROR;
  if (argc > 4 && std::strcmp(argv[4], "logit") == 0) {
    loss_type = LOG_LIKELIHOOD;
  }

  g_conf.loss = loss_type;

  DataVector::iterator iter = d.begin();

  std::string predict_file = input_file + ".predict";
  std::ofstream predict_output(predict_file.c_str());

  Auc auc;
  double sum = 0.0;
  for ( ; iter != d.end(); ++iter) {
    ValueType p;

    if (loss_type == SQUARED_ERROR) {
      p = gbdt.Predict(**iter);
      sum += Squared(p - (*iter)->label) * (*iter)->weight;
    } else if (loss_type == LOG_LIKELIHOOD) {
      p = gbdt.Predict(**iter);
      p = Logit(p);
      auc.Add(p, (*iter)->label);
    }

    predict_output << p << " " << (*iter)->ToString() << std::endl;
  }

  if (loss_type == SQUARED_ERROR) {
    std::cout << "rmse: " << std::sqrt(sum / d.size()) << std::endl;
  } else {
    std::cout << "auc: " << auc.CalculateAuc() << std::endl;
    auc.PrintConfusionTable();
  }

  CleanDataVector(&d);
  FreeVector(&d);
  return 0;
}
