#ifndef LOSS_H
#define LOSS_H

#include "math_util.hpp"
#include <map>

#include <iostream>

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

  bool Register(const std::string &name, CreateFn creater) {
    auto r = creaters_.insert(std::make_pair(name, creater));
    return r.second;
  }

  Objective* Create(const std::string &name) {
    auto iter = creaters_.find(name);
    if (iter == creaters_.end()) {
      return NULL;
    }
    return iter->second();
  }

  void GetAllCandidates(std::vector<std::string> *candidates) {
    candidates->clear();
    for (auto iter = creaters_.begin(); iter != creaters_.end(); ++iter) {
      candidates->push_back(iter->first);
    }
  }

  void PrintAllCandidates() {
    std::cout << "Objective Candidates: ";
    for (auto iter = creaters_.begin(); iter != creaters_.end(); ++iter) {
      std::cout << iter->first << ",";
    }
    std::cout << std::endl;
  }

  static LossFactory* GetInstance() {
    static LossFactory inst;
    return &inst;
  }
 protected:
  LossFactory() {}
 private:
  std::map<std::string, CreateFn> creaters_;
};



#ifndef REGISTER_OBJECTIVE
#define REGISTER_OBJECTIVE(CLZ)                                 \
  inline Objective* create__ ## CLZ() { return new CLZ(); }            \
  void register__ ## CLZ(void) __attribute__ ((constructor));   \
  inline void register__ ## CLZ(void) {                                        \
    LossFactory::GetInstance()->Register(#CLZ, create__ ## CLZ);\
  }
#endif


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

REGISTER_OBJECTIVE(SquaredError)

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

REGISTER_OBJECTIVE(LogLoss)


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

REGISTER_OBJECTIVE(LAD)

}  // gbdt

#endif /* LOSS_H */
