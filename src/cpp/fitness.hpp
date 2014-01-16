// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _FITNESS_H_
#define _FITNESS_H_
#include "data.hpp"

namespace gbdt {
bool AlmostEqual(ValueType v1, ValueType v2);

bool Same(const DataVector &data);
bool Same(const DataVector &data, size_t len);

ValueType Average(const DataVector & data);
ValueType Average(const DataVector & data, size_t len);

bool FindSplit(DataVector *data, int *index, ValueType *value);
bool FindSplit(DataVector *data, size_t len, int *index, ValueType *value);

void SplitData(const DataVector &data, int index, ValueType value, DataVector *output);
void SplitData(const DataVector &data, size_t len, int index, ValueType value, DataVector *output);

double RMSE(const DataVector &data, const PredictVector &predict);
double RMSE(const DataVector &data, const PredictVector &predict, size_t len);

template <typename T>
T Squared(const T &v) {
  return v * v;
}

}

#endif /* _FITNESS_H_ */
