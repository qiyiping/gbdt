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
  Auc(): confusion_table(4, 0L), threshold(0.5F) {}

  void SetThreshold(ValueType t) { threshold = t; }
  void Add(ValueType score, ValueType label);
  ValueType CalculateAuc();
  const std::vector<long> &GetConfusionTable() const { return confusion_table; }
  void PrintConfusionTable() const;

 private:
  std::vector<ValueType> positive_scores;
  std::vector<ValueType> negative_scores;
  std::vector<long> confusion_table;
  ValueType threshold;
};

}

#endif /* _AUC_H_ */
