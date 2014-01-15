// Author: qiyiping@gmail.com (Yiping Qi)
#ifndef _GBDT_H_
#define _GBDT_H_
#include "tree.hpp"

namespace gbdt {
class GBDT {
 public:
  GBDT(): trees(NULL) {}
  void Fit(DataVector *d);
  ValueType Predict(const Tuple &t) const;
 private:
  RegressionTree *trees;
};
}

#endif /* _GBDT_H_ */
