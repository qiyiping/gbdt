// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _DATA_H_
#define _DATA_H_
#include <limits>
#include <string>
#include <vector>
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
      number_of_feature(0),
      label(0),
      target(0),
      weight(0) {}

  ~Tuple() {
    delete[] feature;
  }

  static Tuple* FromString(const std::string &l, int n);

  std::string ToString() const;

 public:
  ValueType *feature;
  ValueType label;
  ValueType target;
  ValueType weight;

 private:
  DISALLOW_COPY_AND_ASSIGN(Tuple);
};

class Configure {
 public:
  int number_of_feature;
  int max_depth;
};

extern Configure gConf;

typedef std::vector<Tuple *> DataVector;
}

#endif /* _DATA_H_ */
