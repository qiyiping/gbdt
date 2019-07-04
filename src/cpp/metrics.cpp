#include "metrics.hpp"
#include "math_util.hpp"
#include "auc.hpp"
namespace gbdt {

double Metrics::MeanAbsoluteError(DataVector &d, PredictVector &p, int len) {
  double s = 0, c = 0;
  for (int i = 0; i < len; ++i) {
    s += Abs(d[i]->label - p[i]) * d[i]->weight;
    c += d[i]->weight;
  }
  return s/c;
}

double Metrics::MeanSquaredError(DataVector &d, PredictVector &p, int len) {
  double s = 0, c = 0;
  for (int i = 0; i < len; ++i) {
    s += Squared(d[i]->label - p[i]) * d[i]->weight;
    c += d[i]->weight;
  }
  return s/c;
}
double Metrics::AucScore(DataVector &d, PredictVector &p, int len) {
  Auc auc;
  for (int i = 0; i < len; ++i) {
    double y = Logit(p[i]);
    auc.Add(y, d[i]->label);
  }

  return auc.CalculateAuc();
}

double Metrics::LogLoss(DataVector &d, PredictVector &p, int len) {
  double s = 0, c = 0;
  for (int i = 0; i < len; ++i) {
    s += 2.0 * std::log(1 + std::exp(-2.0*d[i]->label*p[i])) * d[i]->weight;
    c += d[i]->weight;
  }
  return s/c;
}


}  // gbdt

