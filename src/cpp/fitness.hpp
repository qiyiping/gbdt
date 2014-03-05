// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _FITNESS_H_
#define _FITNESS_H_
#include "data.hpp"
#include <cmath>
#include <cassert>

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

template <typename T>
T Squared(const T &v) {
  return v * v;
}

template <typename T>
T Abs(const T &v) {
  return v >= 0? v : -v;
}

///////////////////////////////////////////////////////////////////////
// Following functions are for gradient boosting classifier
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

}

#endif /* _FITNESS_H_ */
