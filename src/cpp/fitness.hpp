// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _FITNESS_H_
#define _FITNESS_H_
#include "data.hpp"

namespace gbdt {
bool Same(const DataVector &data);
bool AlmostEqual(ValueType v1, ValueType v2);
ValueType Average(const DataVector & data);
bool FindSplit(DataVector *data, int *index, ValueType *value);
void SplitData(const DataVector &data, int index, ValueType value, DataVector *output);

template <typename T>
T Squared(const T &v) {
  return v * v;
}

}

#endif /* _FITNESS_H_ */
