// Author: qiyiping@gmail.com (Yiping Qi)
#ifndef _GBDT_H_
#define _GBDT_H_
#include "tree.hpp"

namespace gbdt {
class GBDT {
 public:
  GBDT(): trees(NULL), bias(0) {}
  void Fit(DataVector *d);
  ValueType Predict(const Tuple &t) const {
    return Predict(t, g_conf.iterations);
  }

  std::string Save() const;
  void Load(const std::string &s);
 private:
  ValueType Predict(const Tuple &t, size_t n) const;
  void Init(const DataVector &d, size_t len);
 private:
  RegressionTree *trees;
  ValueType bias;
};
}

#endif /* _GBDT_H_ */
