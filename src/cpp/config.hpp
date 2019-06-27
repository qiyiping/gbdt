// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <cstddef> // for size_t
#include <string>
#include <memory>
#include <vector>



namespace gbdt {

class Objective;

// Training settings
class Configure {
 public:
  int number_of_feature;      // number of features
  int max_depth;              // max depth for each tree
  int iterations;             // number of trees in gbdt
  double shrinkage;               // shrinkage parameter
  double feature_sample_ratio;    // portion of features to be splited
  double data_sample_ratio;       // portion of data to be fitted in each iteration
  int min_leaf_size;          // min number of nodes in leaf

  std::shared_ptr<Objective> loss; // loss type

  bool debug;                    // show debug info?

  std::vector<double> feature_costs;         // mannually set feature costs in order to tune the model
  bool enable_feature_tunning;   // when set true, `feature_costs' is used to tune the model

  bool enable_initial_guess;

  Configure():
      feature_sample_ratio(1),
      data_sample_ratio(1),
      min_leaf_size(0),
      loss(NULL),
      debug(false),
      enable_feature_tunning(false),
      enable_initial_guess(false) {}

  ~Configure() {}

  bool LoadFeatureCost(const std::string &cost_file);
  void ResetFeatureCost() {
    feature_costs.clear();
    enable_feature_tunning = false;
  }

  std::string ToString() const;
};
}

#endif /* _CONFIG_H_ */
