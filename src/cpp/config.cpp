#include "config.hpp"
#include "util.hpp"
#include <sstream>
#include <fstream>
#include "loss.hpp" // for `Objective'

namespace gbdt {

std::string Configure::ToString() const {
  std::stringstream s;
  s << "number of features = " << number_of_feature << std::endl
    << "min leaf size = " << min_leaf_size << std::endl
    << "maximum depth = " << max_depth << std::endl
    << "iterations = " << iterations << std::endl
    << "shrinkage = " << shrinkage << std::endl
    << "feature sample ratio = " << feature_sample_ratio << std::endl
    << "data sample ratio = " << data_sample_ratio << std::endl
    << "debug enabled = " << debug << std::endl
    << "loss type = " << (loss.get()? loss->GetName() : "NA") << std::endl
    << "feature tuning enabled = " << enable_feature_tunning << std::endl
    << "initial guess enabled = " << enable_initial_guess << std::endl;
  return s.str();
}

// each row of cost file is formated as follows:
// feature_index:feature_cost
// e.g.:
// 1:0.8
//
// fitness of feature is tuned as follows:
// tuned fitness = fitness * feature_cost
bool Configure::LoadFeatureCost(const std::string &cost_file) {
  std::ifstream inputstream(cost_file.c_str());
  if (!inputstream)
    return false;

  feature_costs.clear();
  for (int i = 0; i < number_of_feature; ++i)
    feature_costs.push_back(1.0);

  std::string l;
  while(std::getline(inputstream, l)) {
    if (l.empty() || l[0] == '#')
      continue;
    size_t found = l.find(":");
    int idx = std::stoi(l.substr(0, found));
    double cost = std::stod(l.substr(found+1));
    feature_costs[idx] = cost;
  }

  enable_feature_tunning = true;

  return true;
}

}
