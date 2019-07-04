#ifndef COMMON_LOSS_H
#define COMMON_LOSS_H
#include "loss.hpp"
#include "math_util.hpp"

namespace gbdt {

///////////////////////////////////////////////////////////////////////
// Followings are common objectives, including squared error, log loss
// and least absolute deviation.
//
// For designing your custom objective, you can include `loss.hpp',
// implement `Objective' interface. Please refer to
// `custom_loss_example.hpp' for the details.
///////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////
// Regression
//
// y \in R
//
// Loss function:
// L(y, F(x)) = (y - F(x))^2
//
// Gradient:
// d = 2 \times (F(x) - y)
//
// Optimal terminal node value (approximated):
// rj = sum_{i \in Rj}(di) / |Rj|
///////////////////////////////////////////////////////////////////////

class SquaredError: public Objective {
 public:
  std::string GetName() const {
    return "SquaredError";
  }

  double GetBias(DataVector &d, size_t len) const {
    double s = 0;
    double c = 0;
    for (size_t i = 0; i < len; ++i) {
      s += d[i]->label * d[i]->weight;
      c += d[i]->weight;
    }

    return s / c;
  }

  double GetLoss(const Tuple &t, ValueType f) const {
    return Squared(t.label-f) * 0.5 * t.weight;
  }

  void UpdateGradient(Tuple *t, ValueType f) const {
    t->target = t->label - f;
  }

  double GetRegionPrediction(DataVector &d, size_t len) const {
    return Average(d, len);
  }
};

DECLARE_OBJECTIVE_REGISTRATION(SquaredError)

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

class LogLoss: public Objective {
 public:
  std::string GetName() const {
    return "LogLoss";
  }

  double GetBias(DataVector &d, size_t len) const {
    double s = 0;
    double c = 0;
    for (size_t i = 0; i < len; ++i) {
      s += d[i]->label * d[i]->weight;
      c += d[i]->weight;
    }

    double v = s / c;
    return static_cast<ValueType>(std::log((1+v) / (1-v)) / 2.0);
  }

  double GetLoss(const Tuple &t, ValueType f) const {
    return 2.0 * std::log(1 + std::exp(-2.0*t.label*f));
  }

  void UpdateGradient(Tuple *t, ValueType f) const {
    t->target = 2.0 * t->label / (1 + std::exp(2.0*t->label*f));
  }

  double GetRegionPrediction(DataVector &d, size_t len) const {
    assert(d.size() >= len);

    double s = 0;
    double c = 0;
    for (size_t i = 0; i < len; ++i) {
      s += d[i]->target * d[i]->weight;
      double y = Abs(d[i]->target);
      c += y*(2-y) * d[i]->weight;
    }

    if (c == 0) {
      return static_cast<ValueType>(0);
    } else {
      return static_cast<ValueType> (s / c);
    }
  }
};

DECLARE_OBJECTIVE_REGISTRATION(LogLoss)


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

class LAD: public Objective {
 public:
  std::string GetName() const {
    return "LeastAbsoluteDeviation";
  }

  double GetBias(DataVector &d, size_t len) const {
    return WeightedLabelMedian(d, len);
  }

  double GetLoss(const Tuple &t, ValueType f) const {
    return Abs(t.label-f);
  }

  void UpdateGradient(Tuple *t, ValueType f) const {
    t->residual = t->label - f;
    t->target = Sign(t->residual);
  }

  double GetRegionPrediction(DataVector &d, size_t len) const {
    assert(d.size() >= len);
    return WeightedResidualMedian(d, len);
  }
};

DECLARE_OBJECTIVE_REGISTRATION(LAD)

}  // gbdt

#endif /* COMMON_LOSS_H */
