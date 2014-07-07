// Author: qiyiping@gmail.com (Yiping Qi)

#include "fitness.hpp"
#include "tree.hpp"
#include <algorithm>
#include <iostream>

#ifdef USE_OPENMP
#include <parallel/algorithm>  // openmp
#endif


namespace {
struct TupleCompare {
  TupleCompare(int i): index(i) {}

  bool operator () (const gbdt::Tuple *t1, const gbdt::Tuple *t2) {
    return t1->feature[index] < t2->feature[index];
  }

  int index;
};
}

namespace gbdt {

bool AlmostEqual(ValueType v1, ValueType v2) {
  ValueType diff = Abs(v1-v2);
  if (diff < 1.0e-5)
    return true;
  return false;
}

bool Same(const DataVector &data, size_t len) {
  assert(len <= data.size());
  if (len <= 1)
    return true;

  ValueType t = data[0]->target;
  for (size_t i = 1; i < len; ++i) {
    if (!AlmostEqual(t, data[i]->target))
      return false;
  }
  return true;
}

ValueType Average(const DataVector & data, size_t len) {
  assert(len <= data.size());
  if (len == 0)
    return 0;
  double s = 0;
  double c = 0;
  for (size_t i = 0; i < len; ++i) {
    s += data[i]->target * data[i]->weight;
    c += data[i]->weight;
  }
  return static_cast<ValueType>(s / c);
}

bool FindSplit(DataVector *data, size_t m,
               int *index, ValueType *value, double *gain) {
  size_t n = g_conf.number_of_feature;
  double best_fitness = std::numeric_limits<double>::max();

  std::vector<int> fv;
  for (int i = 0; i < n; ++i) {
    fv.push_back(i);
  }

  size_t fn = n;
  if (g_conf.feature_sample_ratio < 1) {
    fn = static_cast<size_t>(n*g_conf.feature_sample_ratio);
    std::random_shuffle(fv.begin(), fv.end());
  }

  for (size_t k = 0; k < fn; ++k) {
    int i = fv[k];
    ValueType v;
    double impurity;
    double g;
    if (GetImpurity(data, m, i, &v, &impurity, &g)) {
      // Choose feature with smallest impurity to split.  If there's
      // no unknown value, it's equivalent to choose feature with
      // largest gain
      if (best_fitness > impurity) {
        best_fitness = impurity;
        *index = i;
        *value = v;
        *gain = g;
      }
    }
  }

  return best_fitness != std::numeric_limits<double>::max();
}

bool GetImpurity(DataVector *data, size_t len,
                 int index, ValueType *value,
                 double *impurity, double *gain) {
  *impurity = std::numeric_limits<double>::max();
  *value = kUnknownValue;
  *gain = 0;

#ifndef USE_OPENMP
  std::sort(data->begin(), data->begin() + len, TupleCompare(index));
#else
  __gnu_parallel::sort(data->begin(), data->begin() + len, TupleCompare(index));
#endif
  size_t unknown = 0;
  double s = 0;
  double ss = 0;
  double c = 0;

  while (unknown < len && (*data)[unknown]->feature[index] == kUnknownValue) {
    s += (*data)[unknown]->target * (*data)[unknown]->weight;
    ss += Squared((*data)[unknown]->target) * (*data)[unknown]->weight;
    c += (*data)[unknown]->weight;
    unknown++;
  }

  if (unknown == len) {
    return false;
  }

  double fitness0 = c > 1? (ss - s*s/c) : 0;
  if (fitness0 < 0) {
    // std::cerr << "fitness0 < 0: " << fitness0 << std::endl;
    fitness0 = 0;
  }

  s = 0;
  ss = 0;
  c = 0;
  for (size_t j = unknown; j < len; ++j) {
    s += (*data)[j]->target * (*data)[j]->weight;
    ss += Squared((*data)[j]->target) * (*data)[j]->weight;
    c += (*data)[j]->weight;
  }

  double fitness00 = c > 1? (ss - s*s/c) : 0;

  double ls = 0, lss = 0, lc = 0;
  double rs = s, rss = ss, rc = c;
  double fitness1 = 0, fitness2 = 0;
  for (size_t j = unknown; j < len-1; ++j) {
    s = (*data)[j]->target * (*data)[j]->weight;
    ss = Squared((*data)[j]->target) * (*data)[j]->weight;
    c = (*data)[j]->weight;

    ls += s;
    lss += ss;
    lc += c;

    rs -= s;
    rss -= ss;
    rc -= c;

    ValueType f1 = (*data)[j]->feature[index];
    ValueType f2 = (*data)[j+1]->feature[index];
    if (AlmostEqual(f1, f2))
      continue;

    fitness1 = lc > 1? (lss - ls*ls/lc) : 0;
    if (fitness1 < 0) {
      // std::cerr << "fitness1 < 0: " << fitness1 << std::endl;
      fitness1 = 0;
    }

    fitness2 = rc > 1? (rss - rs*rs/rc) : 0;
    if (fitness2 < 0) {
      // std::cerr << "fitness2 < 0: " << fitness2 << std::endl;
      fitness2 = 0;
    }

    double fitness = fitness0 + fitness1 + fitness2;

    if (g_conf.feature_costs && g_conf.enable_feature_tunning) {
      fitness *= g_conf.feature_costs[index];
    }

    if (*impurity > fitness) {
      *impurity = fitness;
      *value = (f1+f2)/2;
      *gain = fitness00 - fitness1 - fitness2;
    }
  }

  return *impurity != std::numeric_limits<double>::max();
}

void SplitData(const DataVector &data, size_t len, int index, ValueType value, DataVector *output) {
  for (size_t i = 0; i < len; ++i) {
    if (data[i]->feature[index] == kUnknownValue) {
      output[Node::UNKNOWN].push_back(data[i]);
    } else if (data[i]->feature[index] < value) {
      output[Node::LT].push_back(data[i]);
    } else {
      output[Node::GE].push_back(data[i]);
    }
  }
}

double RMSE(const DataVector &data, const PredictVector &predict, size_t len) {
  assert(data.size() >= len);
  assert(predict.size() >= len);
  double s = 0;
  double c = 0;

  for (size_t i = 0; i < data.size(); ++i) {
    s += Squared(predict[i] - data[i]->label) * data[i]->weight;
    c += data[i]->weight;
  }

  return std::sqrt(s / c);
}

ValueType LogitOptimalValue(const DataVector &d, size_t len) {
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

}
