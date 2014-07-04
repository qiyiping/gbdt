// Author: qiyiping@gmail.com (Yiping Qi)
#include "gbdt.hpp"
#include "fitness.hpp"
#include "util.hpp"
#include "auc.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <boost/lexical_cast.hpp>

#ifdef USE_OPENMP
#include <parallel/algorithm>  // openmp
#endif

namespace gbdt {
ValueType GBDT::Predict(const Tuple &t, size_t n) const {
  if (!trees)
    return kUnknownValue;

  assert(n <= iterations);

  ValueType r = bias;
  if (g_conf.enable_initial_guess) {
    r = t.initial_guess;
  }

  for (size_t i = 0; i < n; ++i) {
    r += shrinkage * trees[i].Predict(t);
  }

  return r;
}

ValueType GBDT::Predict(const Tuple &t, size_t n, double *p) const {
  if (!trees)
    return kUnknownValue;

  assert(n <= iterations);

  ValueType r = bias;
  if (g_conf.enable_initial_guess) {
    r = t.initial_guess;
  }

  for (size_t i = 0; i < n; ++i) {
    r += shrinkage * trees[i].Predict(t, p);
  }

  return r;
}

void GBDT::Init(const DataVector &d, size_t len) {
  assert(d.size() >= len);

  if (g_conf.enable_initial_guess) {
    return;
  }

  double s = 0;
  double c = 0;
  for (size_t i = 0; i < len; ++i) {
    s += d[i]->label * d[i]->weight;
    c += d[i]->weight;
  }

  double v = s / c;

  if (g_conf.loss == SQUARED_ERROR) {
    bias = static_cast<ValueType>(v);
  } else if (g_conf.loss == LOG_LIKELIHOOD) {
    bias = static_cast<ValueType>(std::log((1+v) / (1-v)) / 2.0);
  }
}

void GBDT::Fit(DataVector *d) {
  delete[] trees;
  trees = new RegressionTree[g_conf.iterations];

  size_t samples = d->size();
  if (g_conf.data_sample_ratio < 1) {
    samples = static_cast<size_t>(d->size() * g_conf.data_sample_ratio);
  }

  Init(*d, d->size());

  for (size_t i = 0; i < g_conf.iterations; ++i) {
    std::cout  << "iteration: " << i << std::endl;

    if (samples < d->size()) {
#ifndef USE_OPENMP
      std::random_shuffle(d->begin(), d->end());
#else
      __gnu_parallel::random_shuffle(d->begin(), d->end());
#endif
    }

    if (g_conf.loss == SQUARED_ERROR) {
      for (size_t j = 0; j < samples; ++j) {
        ValueType p = Predict(*(*d)[j], i);
        (*d)[j]->target = (*d)[j]->label - p;
      }

      if (g_conf.debug) {
        double s = 0;
        double c = 0;
        DataVector::iterator iter = d->begin();
        for ( ; iter != d->end(); ++iter) {
          ValueType p = Predict(**iter, i);
          s += Squared((*iter)->label - p) * (*iter)->weight;
          c += (*iter)->weight;
        }
        std::cout << "rmse: " << std::sqrt(s / c) << std::endl;
      }
    } else if (g_conf.loss == LOG_LIKELIHOOD) {
      for (size_t j = 0; j < samples; ++j) {
        ValueType p = Predict(*(*d)[j], i);
        (*d)[j]->target =
            static_cast<ValueType>(LogitLossGradient((*d)[j]->label, p));
      }

      if (g_conf.debug) {
        Auc auc;
        DataVector::iterator iter = d->begin();
        for ( ; iter != d->end(); ++iter) {
          ValueType p = Logit(Predict(**iter, i));
          auc.Add(p, (*iter)->label);
        }
        std::cout << "auc: " << auc.CalculateAuc() << std::endl;
      }
    }

    trees[i].Fit(d, samples);
  }


  // Calculate gain
  delete[] gain;
  gain = new double[g_conf.number_of_feature];

  for (size_t i = 0; i < g_conf.number_of_feature; ++i) {
    gain[i] = 0.0;
  }

  for (size_t j = 0; j < iterations; ++j) {
    double *g = trees[j].GetGain();
    for (size_t i = 0; i < g_conf.number_of_feature; ++i) {
      gain[i] += g[i];
    }
  }
}

std::string GBDT::Save() const {
  std::vector<std::string> vs;
  vs.push_back(boost::lexical_cast<std::string>(shrinkage));
  vs.push_back(boost::lexical_cast<std::string>(bias));
  for (size_t i = 0; i < iterations; ++i) {
    vs.push_back(trees[i].Save());
  }
  return JoinString(vs, "\n;\n");
}

void GBDT::Load(const std::string &s) {
  delete[] trees;
  std::vector<std::string> vs;
  SplitString(s, "\n;\n", &vs);

  iterations = vs.size() - 2;
  shrinkage = boost::lexical_cast<ValueType>(vs[0]);
  bias = boost::lexical_cast<ValueType>(vs[1]);

  trees = new RegressionTree[iterations];
  for (size_t i = 0; i < iterations; ++i) {
    trees[i].Load(vs[i+2]);
  }
}

GBDT::~GBDT() {
  delete[] trees;
  delete[] gain;
}
}
