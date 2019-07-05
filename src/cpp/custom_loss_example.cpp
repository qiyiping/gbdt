#include "custom_loss_example.hpp"

namespace {
struct ResidualCompare {
  bool operator () (const gbdt::Tuple *t1, const gbdt::Tuple *t2) {
    return t1->residual < t2->residual;
  }
};
}

namespace gbdt {

double MyLoss::GetRegionPrediction(DataVector &d, size_t len) const {
  std::sort(d.begin(), d.begin()+len, ResidualCompare());
  double n = 0, m = 0;
  int k = 0;
  for (k = 0; k < len; ++k) {
    if (d[k]->label >= 0) {
      n += d[k]->weight;
    } else {
      m += d[k]->weight;
    }
  }

  double r = 0.0;
  double i = 0, j = 0;
  for (k = 0; k < len; ++k) {
    if (d[k]->label >= 0) {
      i += d[k]->weight;
    } else {
      j += d[k]->weight;
    }

    if (j+2*i-m-n > 0) {
      break;
    }
  }

  if (k >= len) {
    r = d[k-1]->residual;
  } else if (k-1 >= 0) {
    r = (d[k]->residual + d[k-1]->residual) / 2.0;
  } else {
    r = d[k]->residual;
  }

  return r;
}

DEFINE_OBJECTIVE_CREATOR(MyLoss)

SHARED_LIB_INIT_BEGIN
SHARED_LIB_INIT_REGISTER(MyLoss)
SHARED_LIB_INIT_END

}  // gbdt
