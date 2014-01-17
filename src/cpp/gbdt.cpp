// Author: qiyiping@gmail.com (Yiping Qi)
#include "gbdt.hpp"
#include "fitness.hpp"
#include "util.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>

#ifdef USE_OPENMP
#include <parallel/algorithm>  // openmp
#endif

namespace gbdt {
ValueType GBDT::Predict(const Tuple &t) const {
  if (!trees)
    return kUnknownValue;

  ValueType r = 0;
  for (size_t i = 0; i < g_conf.iterations; ++i) {
    r += g_conf.shrinkage * trees[i].Predict(t);
  }

  return r;
}

void GBDT::Fit(DataVector *d) {
  delete[] trees;
  trees = new RegressionTree[g_conf.iterations];

  size_t samples = d->size();
  if (g_conf.data_sample_ratio < 1) {
    samples = static_cast<size_t>(d->size() * g_conf.data_sample_ratio);
  }

  for (size_t i = 0; i < g_conf.iterations; ++i) {
    if (samples < d->size()) {
#ifndef USE_OPENMP
      std::random_shuffle(d->begin(), d->end());
#else
      __gnu_parallel::random_shuffle(d->begin(), d->end());
#endif
    }

    trees[i].Fit(d, samples);

    DataVector::iterator iter = d->begin();
    double s = 0;
    double c = 0;
    for ( ; iter != d->end(); ++iter) {
      ValueType p = trees[i].Predict(**iter);
      (*iter)->target -= g_conf.shrinkage * p;
      s += Squared((*iter)->target) * (*iter)->weight;
      c += (*iter)->weight;
    }

    if (g_conf.debug) {
      std::cout  << "iteration: " << i << std::endl
                 << "rmse: " << std::sqrt(s / c) << std::endl;
    }
  }
}

std::string GBDT::Save() const {
  std::vector<std::string> vs;
  for (size_t i = 0; i < g_conf.iterations; ++i) {
    vs.push_back(trees[i].Save());
  }
  return JoinString(vs, "\n;\n");
}

void GBDT::Load(const std::string &s) {
  delete[] trees;
  trees = new RegressionTree[g_conf.iterations];
  std::vector<std::string> vs;
  SplitString(s, "\n;\n", &vs);
  size_t j = 0;
  for (size_t i = 0; i < vs.size(); ++i) {
    if (vs[i].empty()) continue;

    trees[j++].Load(vs[i]);
  }

  assert(j == g_conf.iterations);
}
}
