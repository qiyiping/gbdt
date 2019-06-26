// Author: qiyiping@gmail.com (Yiping Qi)

#include "math_util.hpp"
#include "tree.hpp"
#include <algorithm>
#include <iostream>

#ifdef USE_OPENMP
#include <parallel/algorithm>  // openmp
#endif


namespace {
struct ResidualCompare {
  bool operator () (const gbdt::Tuple *t1, const gbdt::Tuple *t2) {
    return t1->residual < t2->residual;
  }
};

struct LabelCompare {
  bool operator () (const gbdt::Tuple *t1, const gbdt::Tuple *t2) {
    return t1->label < t2->label;
  }
};

}

namespace gbdt {

bool AlmostEqual(ValueType v1, ValueType v2) {
  ValueType diff = Abs(v1-v2);
  if (diff < 1.0e-5)
    return true;
  return false;
}

bool Same(const DataVector &data, size_t len) {
  assert(len <= data.size());
  if (len <= 1)
    return true;

  ValueType t = data[0]->target;
  for (size_t i = 1; i < len; ++i) {
    if (!AlmostEqual(t, data[i]->target))
      return false;
  }
  return true;
}

ValueType Average(const DataVector & data, size_t len) {
  assert(len <= data.size());
  if (len == 0)
    return 0;
  double s = 0;
  double c = 0;
  for (size_t i = 0; i < len; ++i) {
    s += data[i]->target * data[i]->weight;
    c += data[i]->weight;
  }
  return static_cast<ValueType>(s / c);
}

double RMSE(const DataVector &data, const PredictVector &predict, size_t len) {
  assert(data.size() >= len);
  assert(predict.size() >= len);
  double s = 0;
  double c = 0;

  for (size_t i = 0; i < data.size(); ++i) {
    s += Squared(predict[i] - data[i]->label) * data[i]->weight;
    c += data[i]->weight;
  }

  return std::sqrt(s / c);
}

double MAE(const DataVector &data, const PredictVector &predict, size_t len) {
  assert(data.size() >= len);
  assert(predict.size() >= len);
  double s = 0;
  double c = 0;

  for (size_t i = 0; i < data.size(); ++i) {
    s += Abs(predict[i] - data[i]->label) * data[i]->weight;
    c += data[i]->weight;
  }

  return s / c;
}

ValueType WeightedResidualMedian(DataVector &d, size_t len) {
  assert(d.size() >= len);
  // simplest implementation using sorting
  // sophisticated approch to find the weighted median is selection algorithm(partition algorithm).
  std::sort(d.begin(), d.begin() + len, ResidualCompare());
  double all_weight = 0.0;
  for (size_t i = 0; i < len; ++i) {
    all_weight += d[i]->weight;
  }

  ValueType weighted_median = 0.0;
  double weight = 0.0;
  for (int i = 0; i < len; ++i) {
    weight += d[i]->weight;
    if (weight * 2 > all_weight) {
      if (i-1 >= 0) {
        weighted_median = (d[i]->residual + d[i-1]->residual) / 2.0;
      } else {
        weighted_median = d[i]->residual;
      }

      break;
    }
  }

  return weighted_median;
}

ValueType WeightedLabelMedian(DataVector &d, size_t len) {
  assert(d.size() >= len);
  // simplest implementation using sorting
  // sophisticated approch to find the weighted median is selection algorithm(partition algorithm).
  std::sort(d.begin(), d.begin() + len, LabelCompare());
  double all_weight = 0.0;
  for (size_t i = 0; i < len; ++i) {
    all_weight += d[i]->weight;
  }

  ValueType weighted_median = 0.0;
  double weight = 0.0;
  for (int i = 0; i < len; ++i) {
    weight += d[i]->weight;
    if (weight * 2 > all_weight) {
      if (i-1 >= 0) {
        weighted_median = (d[i]->label + d[i-1]->label) / 2.0;
      } else {
        weighted_median = d[i]->label;
      }
      break;
    }
  }

  return weighted_median;
}

}
