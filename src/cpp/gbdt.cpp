// Author: qiyiping@gmail.com (Yiping Qi)
#include "gbdt.hpp"
#include "fitness.hpp"
#include <iostream>
#include <cmath>

namespace gbdt {
ValueType GBDT::Predict(const Tuple &t) const {
  if (!trees)
    return kUnknownValue;

  ValueType r = 0;
  for (int i = 0; i < gConf.iterations; ++i) {
    r += gConf.shrinkage * trees[i].Predict(t);
  }

  return r;
}

void GBDT::Fit(DataVector *d) {
  trees = new RegressionTree[gConf.iterations];
  for (int i = 0; i < gConf.iterations; ++i) {
    trees[i].Fit(d);

    DataVector::iterator iter = d->begin();
    ValueType s = 0;
    for ( ; iter != d->end(); ++iter) {
      ValueType p = trees[i].Predict(**iter);
      (*iter)->target -= gConf.shrinkage * p;
      s += Squared((*iter)->target);
    }

    std::cout  << "iteration: " << i << std::endl
               << "rmse: " << std::sqrt(s/d->size()) << std::endl;
  }
}
}
