#ifndef CUSTOM_LOSS_EXAMPLE_H
#define CUSTOM_LOSS_EXAMPLE_H

#include "loss.hpp"
#include "math_util.hpp"

namespace gbdt {

/*
 * y \in R
 * L(y, F(x)) = |y-F(x)| for samples in group A
 *            = max(y-F(x), 0) for samples in group B
 * The sign of y is used as a indicator of the group.
 */
class MyLoss: public Objective {
 public:
  double GetBias(DataVector &d, size_t len) const {
    return 0.0;
  }
  double GetLoss(const Tuple& t, ValueType f) const {
    if (t.label >= 0) {
      return Abs(t.label - f);
    } else {
      return std::max(0.0, -t.label - f);
    }
  }

  void UpdateGradient(Tuple* t, ValueType f) const {
    if (t->label >= 0) {
      t->residual = t->label - f;
      t->target = Sign(t->residual);
    } else {
      t->residual = -t->label - f;
      t->target = t->residual > 0? 1:0;
    }
  }
  double GetRegionPrediction(DataVector &d, size_t len) const;
  std::string GetName() const {
    return "MyLoss";
  }
};

SHARED_LIB_INIT_DECLARE

}  // gbdt

#endif /* CUSTOM_LOSS_EXAMPLE_H */
