// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <cstddef> // for size_t
#include <string>


namespace gbdt {

enum Loss {
  SQUARED_ERROR,                 // regression with squared error loss
  LOG_LIKELIHOOD                 // two-class logistic regression and classification
};

// Training settings
class Configure {
 public:
  size_t number_of_feature;      // number of features
  size_t max_depth;              // max depth for each tree
  size_t iterations;             // number of trees in gbdt
  float shrinkage;               // shrinkage parameter
  float feature_sample_ratio;    // portion of features to be splited
  float data_sample_ratio;       // portion of data to be fitted in each iteration
  size_t min_leaf_size;          // min number of nodes in leaf

  Loss loss;                     // loss type

  bool debug;                    // show debug info?
  bool absolute_gain;            // absolute_gain of gain, default gain.

  double *feature_costs;         // mannually set feature costs in order to tune the model
  bool enable_feature_tunning;   // when set true, `feature_costs' is used to tune the model

  Configure():
      feature_sample_ratio(1),
      data_sample_ratio(1),
      min_leaf_size(0),
      loss(SQUARED_ERROR),
      debug(false),
      feature_costs(NULL),
      enable_feature_tunning(false),
      absolute_gain(false) {}

  ~Configure() { delete[] feature_costs; }

  bool LoadFeatureCost(const std::string &cost_file);
  void ResetFeatureCost() {
    delete[] feature_costs;
    feature_costs = NULL;
    enable_feature_tunning = false;
  }

  std::string ToString() const;
};

extern Configure g_conf;
}

#endif /* _CONFIG_H_ */
