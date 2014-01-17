#include "config.hpp"
#include <sstream>

namespace gbdt {
Configure g_conf;

std::string Configure::ToString() const {
  std::stringstream s;
  s << "number of features = " << number_of_feature << std::endl
    << "min leaf size = " << min_leaf_size << std::endl
    << "maximum depth = " << max_depth << std::endl
    << "iterations = " << iterations << std::endl
    << "shrinkage = " << shrinkage << std::endl
    << "feature sample ratio = " << feature_sample_ratio << std::endl
    << "data sample ratio = " << data_sample_ratio << std::endl;
  return s.str();
}
}
