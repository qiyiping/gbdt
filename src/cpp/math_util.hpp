// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _MATH_UTIL_H_
#define _MATH_UTIL_H_
#include <cmath>
#include <cassert>
#include <algorithm>

#include "data.hpp"

namespace gbdt {
bool AlmostEqual(ValueType v1, ValueType v2);
bool Same(const DataVector &data, size_t len);
ValueType Average(const DataVector &data, size_t len);
double RMSE(const DataVector &data, const PredictVector &predict, size_t len);
double MAE(const DataVector &data, const PredictVector &predict, size_t len);

ValueType WeightedResidualMedian(DataVector &d, size_t len);
ValueType WeightedLabelMedian(DataVector &d, size_t len);

inline
double Logit(ValueType f) {
  return 1.0 / (1 + std::exp(-2.0*f));
}

inline
bool Same(const DataVector &data) {
  return Same(data, data.size());
}

inline
ValueType Average(const DataVector & data) {
  return Average(data, data.size());
}


inline
double RMSE(const DataVector &data, const PredictVector &predict) {
  assert(data.size() == predict.size());
  return RMSE(data, predict, data.size());
}

inline
double MAE(const DataVector &data, const PredictVector &predict) {
  assert(data.size() == predict.size());
  return MAE(data, predict, data.size());
}

template <typename T>
T Squared(const T &v) {
  return v * v;
}

template <typename T>
T Abs(const T &v) {
  return v >= 0? v : -v;
}

template <typename T>
T Sign(const T &v) {
  return v > 0? 1 : -1;
}

template <typename T, typename Compare>
T Median(std::vector<T> &data, size_t len, Compare comp) {
  assert(data.size() > len);
  std::nth_element(data.begin(),
                   data.begin() + len / 2,
                   data.begin() + len,
                   comp);
  return data[len / 2];
}

}

#endif /* _MATH_UTIL_H_ */
