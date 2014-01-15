// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _DATA_H_
#define _DATA_H_
#include <limits>
#include <string>
#include <vector>
#include "config.hpp"
#include "util.hpp"

namespace gbdt {
typedef float ValueType;
const ValueType kValueTypeMax = std::numeric_limits<ValueType>::max();
const ValueType kValueTypeMin = std::numeric_limits<ValueType>::min();
const ValueType kUnknownValue = kValueTypeMin;

class Tuple {
 public:
  Tuple():
      feature(NULL),
      label(0),
      target(0),
      weight(0) {}

  ~Tuple() {
    delete[] feature;
  }

  static Tuple* FromString(const std::string &l);

  std::string ToString() const;

 public:
  ValueType *feature;
  ValueType label;
  ValueType target;
  ValueType weight;

 private:
  DISALLOW_COPY_AND_ASSIGN(Tuple);
};

typedef std::vector<Tuple *> DataVector;
void CleanDataVector(DataVector *data);
bool LoadDataFromFile(const std::string &path, DataVector *data);

typedef std::vector<ValueType> PredictVector;
}

#endif /* _DATA_H_ */
