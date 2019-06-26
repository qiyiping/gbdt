// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _DATA_H_
#define _DATA_H_
#include <limits>
#include <string>
#include <vector>
#include "util.hpp"

namespace gbdt {
// typedef float ValueType;
typedef double ValueType;
const ValueType kValueTypeMax = std::numeric_limits<ValueType>::max();
const ValueType kValueTypeMin = std::numeric_limits<ValueType>::min();
const ValueType kUnknownValue = kValueTypeMin;

// enum VariableType {
//   CONTINUOUS,
//   ORDINAL,
//   NOMINAL
// };

class Tuple {
 public:
  Tuple():
      feature(NULL),
      label(0),
      target(0),
      weight(0),
      residual(0),
      initial_guess(kUnknownValue) {}

  ~Tuple() {
    delete[] feature;
  }

  static Tuple* FromString(const std::string &l,
                           int number_of_feature,
                           bool two_class_classification,
                           bool load_initial_guess=false);

  std::string ToString(int number_of_feature,
                       bool output_initial_guess=false) const;

 public:
  ValueType *feature;
  ValueType label;
  ValueType target;
  ValueType weight;
  ValueType residual;

  ValueType initial_guess;

 private:
  DISALLOW_COPY_AND_ASSIGN(Tuple);
};

typedef std::vector<Tuple *> DataVector;
void CleanDataVector(DataVector *data);
bool LoadDataFromFile(const std::string &path,
                      DataVector *data,
                      int number_of_feature,
                      bool two_class_classification,
                      bool load_initial_guess=false,
                      bool ignore_weight=false);

typedef std::vector<ValueType> PredictVector;
}

#endif /* _DATA_H_ */
