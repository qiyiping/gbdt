// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _FITNESS_H_
#define _FITNESS_H_
#include "data.hpp"
#include <cmath>
#include <cassert>
#include <algorithm>

namespace gbdt {
bool AlmostEqual(ValueType v1, ValueType v2);
bool Same(const DataVector &data, size_t len);
ValueType Average(const DataVector &data, size_t len);
bool FindSplit(DataVector *data, size_t len,
               int *index, ValueType *value, double *gain);
bool GetImpurity(DataVector *data, size_t len,
                 int index, ValueType *value,
                 double *impurity, double *gain);
void SplitData(const DataVector &data, size_t len, int index, ValueType value, DataVector *output);
double RMSE(const DataVector &data, const PredictVector &predict, size_t len);
double MAE(const DataVector &data, const PredictVector &predict, size_t len);

inline
bool Same(const DataVector &data) {
  return Same(data, data.size());
}

inline
ValueType Average(const DataVector & data) {
  return Average(data, data.size());
}

inline
bool FindSplit(DataVector *data, int *index,
               ValueType *value, double *gain) {
  return FindSplit(data, data->size(), index, value, gain);
}

inline
void SplitData(const DataVector &data, int index, ValueType value, DataVector *output) {
  SplitData(data, data.size(), index, value, output);
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

///////////////////////////////////////////////////////////////////////
// Following functions are for gradient boosting classifier
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// Two class classification
//
// y \in {-1, 1}
// Pr(y=1) = 1/(1+exp(-2F(x)))
//
// Loss function:
// L(y, F(x)) = log(1+2exp(-2yF(x)))
//
// Gradient:
// d = 2y/(1+exp(2yF(x)))
//
// Optimal terminal node value (approximated):
// rj = sum_{i \in Rj}(di) / sum_{i \in Rj}(|di|(2-|di|))
///////////////////////////////////////////////////////////////////////


inline
double Logit(ValueType f) {
  return 1.0 / (1 + std::exp(-2.0*f));
}

inline
double LogitLoss(ValueType y, ValueType f) {
  return 2.0 * std::log(1 + std::exp(-2.0*y*f));
}

inline
double LogitLossGradient(ValueType y, ValueType f) {
  return 2.0 * y / (1 + std::exp(2.0*y*f));
}


ValueType LogitOptimalValue(const DataVector &d, size_t len);


///////////////////////////////////////////////////////////////////////
// LAD
//
// y \in R
//
// Loss function:
// L(y, F(x)) = |y-F(x)|
//
// Gradient:
// d = sign(F(x)-y)
//
// Optimal terminal node value (approximated):
// rj = median({y_ji-F(x_ji)})
///////////////////////////////////////////////////////////////////////

inline
double LADLoss(ValueType y, ValueType f) {
  return Abs(y-f);
}

inline
double LADLossGradient(ValueType y, ValueType f) {
  return Sign(y-f);
}

ValueType WeightedResidualMedian(DataVector &d, size_t len);

ValueType WeightedLabelMedian(DataVector &d, size_t len);

inline
ValueType LADOptimalValue(DataVector &d, size_t len) {
  assert(d.size() >= len);
  return WeightedResidualMedian(d, len);
}

}

#endif /* _FITNESS_H_ */
