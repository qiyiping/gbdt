#ifndef METRICS_H
#define METRICS_H

#include <map>
#include <string>
#include "data.hpp"

namespace gbdt {

class Metrics {
 public:
  static double MeanAbsoluteError(DataVector &d, PredictVector &p, int len);
  static double MeanSquaredError(DataVector &d, PredictVector &p, int len);
  static double AucScore(DataVector &d, PredictVector &p, int len);
  static double LogLoss(DataVector &d, PredictVector &p, int len);
};

}  // gbdt

#endif /* METRICS_H */
