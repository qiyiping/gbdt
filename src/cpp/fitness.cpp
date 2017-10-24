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

struct ResidualCompare {
  bool operator () (const gbdt::Tuple *t1, const gbdt::Tuple *t2) {
    return t1->residual < t2->residual;
  }
};

struct LabelCompare {
  bool operator () (const gbdt::Tuple *t1, const gbdt::Tuple *t2) {
    return t1->label < t2->label;
  }
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

  ValueType *v = new ValueType[fn];
  double *impurity = new double[fn];
  double *g = new double[fn];

  #ifdef USE_OPENMP
  #pragma omp parallel for
  #endif
  for (size_t k = 0; k < fn; ++k) {
    GetImpurity(data, m, fv[k], &v[k], &impurity[k], &g[k]);
  }

  for (size_t k = 0; k < fn; ++k) {
    // Choose feature with smallest impurity to split.  If there's
    // no unknown value, it's equivalent to choose feature with
    // largest gain
    if (best_fitness > impurity[k]) {
      best_fitness = impurity[k];
      *index = fv[k];
      *value = v[k];
      *gain = g[k];
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

  DataVector data_copy = DataVector(data->begin(), data->end());

  std::sort(data_copy.begin(), data_copy.begin() + len, TupleCompare(index));

  size_t unknown = 0;
  double s = 0;
  double ss = 0;
  double c = 0;

  while (unknown < len && data_copy[unknown]->feature[index] == kUnknownValue) {
    s += data_copy[unknown]->target * data_copy[unknown]->weight;
    ss += Squared(data_copy[unknown]->target) * data_copy[unknown]->weight;
    c += data_copy[unknown]->weight;
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
    s += data_copy[j]->target * data_copy[j]->weight;
    ss += Squared(data_copy[j]->target) * data_copy[j]->weight;
    c += data_copy[j]->weight;
  }

  double fitness00 = c > 1? (ss - s*s/c) : 0;

  double ls = 0, lss = 0, lc = 0;
  double rs = s, rss = ss, rc = c;
  double fitness1 = 0, fitness2 = 0;
  for (size_t j = unknown; j < len-1; ++j) {
    s = data_copy[j]->target * data_copy[j]->weight;
    ss = Squared(data_copy[j]->target) * data_copy[j]->weight;
    c = data_copy[j]->weight;

    ls += s;
    lss += ss;
    lc += c;

    rs -= s;
    rss -= ss;
    rc -= c;

    ValueType f1 = data_copy[j]->feature[index];
    ValueType f2 = data_copy[j+1]->feature[index];
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

double MAE(const DataVector &data, const PredictVector &predict, size_t len) {
  assert(data.size() >= len);
  assert(predict.size() >= len);
  double s = 0;
  double c = 0;

  for (size_t i = 0; i < data.size(); ++i) {
    s += Abs(predict[i] - data[i]->label) * data[i]->weight;
    c += data[i]->weight;
  }

  return s / c;
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

ValueType WeightedResidualMedian(DataVector &d, size_t len) {
  assert(d.size() >= len);
  // simplest implementation using sorting
  // sophisticated approch to find the weighted median is selection algorithm(partition algorithm).
  std::sort(d.begin(), d.begin() + len, ResidualCompare());
  double all_weight = 0.0;
  for (size_t i = 0; i < len; ++i) {
    all_weight += d[i]->weight;
  }

  ValueType weighted_median = 0.0;
  double weight = 0.0;
  for (int i = 0; i < len; ++i) {
    weight += d[i]->weight;
    if (weight * 2 > all_weight) {
      if (i-1 >= 0) {
        weighted_median = (d[i]->residual + d[i-1]->residual) / 2.0;
      } else {
        weighted_median = d[i]->residual;
      }

      break;
    }
  }

  return weighted_median;
}

ValueType WeightedLabelMedian(DataVector &d, size_t len) {
  assert(d.size() >= len);
  // simplest implementation using sorting
  // sophisticated approch to find the weighted median is selection algorithm(partition algorithm).
  std::sort(d.begin(), d.begin() + len, LabelCompare());
  double all_weight = 0.0;
  for (size_t i = 0; i < len; ++i) {
    all_weight += d[i]->weight;
  }

  ValueType weighted_median = 0.0;
  double weight = 0.0;
  for (int i = 0; i < len; ++i) {
    weight += d[i]->weight;
    if (weight * 2 > all_weight) {
      if (i-1 >= 0) {
        weighted_median = (d[i]->label + d[i-1]->label) / 2.0;
      } else {
        weighted_median = d[i]->label;
      }
      break;
    }
  }

  return weighted_median;
}

}
