#ifndef CUSTOM_LOSS_EXAMPLE_H
#define CUSTOM_LOSS_EXAMPLE_H

#include "loss.hpp"

namespace gbdt {

class MyLoss: public Objective {
 public:
  double GetBias(DataVector &d, size_t len) const {
    return 0.0;
  }
  double GetLoss(const Tuple& t, ValueType f) const {
    return 0.0;
  }
  void UpdateGradient(Tuple* t, ValueType f) const {
    // TBD
  }
  double GetRegionPrediction(DataVector &d, size_t len) const;
  std::string GetName() const {
    return "MyLoss";
  }
};

SHARED_LIB_INIT_DECLARE

}  // gbdt

#endif /* CUSTOM_LOSS_EXAMPLE_H */
