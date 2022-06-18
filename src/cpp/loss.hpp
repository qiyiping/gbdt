#ifndef LOSS_H
#define LOSS_H

#include <map>
#include "data.hpp"
#include "math_util.hpp"

namespace gbdt {

///////////////////////////////////////////////////////////////////////
// Objective for function approximation
///////////////////////////////////////////////////////////////////////
class Objective {
 public:
  virtual double GetBias(DataVector &d, size_t len) const = 0;
  virtual double GetLoss(const Tuple& t, ValueType f) const = 0;
  virtual void UpdateGradient(Tuple* t, ValueType f) const = 0;
  virtual double GetRegionPrediction(DataVector &d, size_t len) const = 0;
  virtual std::string GetName() const = 0;
  virtual ~Objective() {}
};

class LossFactory {
 public:
  using CreateFn = Objective* (*) ();

  bool Register(const std::string &name, CreateFn creater);
  Objective* Create(const std::string &name);
  void GetAllCandidates(std::vector<std::string> *candidates);
  void PrintAllCandidates();
  bool LoadSharedLib(const std::string &path);
  ~LossFactory();
  static LossFactory* GetInstance();
 protected:
  LossFactory() {}
 private:
  std::map<std::string, CreateFn> creaters_;
  std::vector<void*> handles_;
};


#ifndef DECLARE_OBJECTIVE_REGISTRATION
#define DECLARE_OBJECTIVE_REGISTRATION(CLZ)                     \
  void register__ ## CLZ(void) __attribute__ ((constructor));
#endif

#ifndef DEFINE_OBJECTIVE_CREATOR
#define DEFINE_OBJECTIVE_CREATOR(CLZ)                           \
  static Objective* create__ ## CLZ() { return new CLZ(); }
#endif

#ifndef DEFINE_OBJECTIVE_REGISTRATION
#define DEFINE_OBJECTIVE_REGISTRATION(CLZ)                              \
  DEFINE_OBJECTIVE_CREATOR(CLZ)                                         \
  void register__ ## CLZ(void) {                                        \
    LossFactory::GetInstance()->Register(#CLZ, create__ ## CLZ);        \
  }
#endif

#ifndef SHARED_LIB_INIT_DECLARE
#define SHARED_LIB_INIT_DECLARE                 \
  extern "C" void init(LossFactory *factory);
#endif

#ifndef SHARED_LIB_INIT_BEGIN
#define SHARED_LIB_INIT_BEGIN                   \
  void init(LossFactory *factory) {
#endif

#ifndef SHARED_LIB_INIT_END
#define SHARED_LIB_INIT_END                     \
  }
#endif


#ifndef SHARED_LIB_INIT_REGISTER
#define SHARED_LIB_INIT_REGISTER(CLZ)           \
  factory->Register(#CLZ, create__ ## CLZ);
#endif




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

#endif /* LOSS_H */
