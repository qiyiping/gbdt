// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _AUC_H_
#define _AUC_H_
#include "data.hpp"
#include <vector>
#include <algorithm>

namespace gbdt {
// imported from mahout:
// org.apache.mahout.classifier.evaluation.Auc

class Auc {
 public:
  Auc() {}

  void Add(ValueType score, ValueType label);
  ValueType CalculateAuc();

 private:
  std::vector<ValueType> positive_scores;
  std::vector<ValueType> negative_scores;
};

}

#endif /* _AUC_H_ */
