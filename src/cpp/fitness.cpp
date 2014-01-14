// Author: qiyiping@gmail.com (Yiping Qi)

#include "fitness.hpp"
#include "tree.hpp"
#include <algorithm>
#include <cassert>

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
  ValueType diff = v1 > v2? (v1-v2) : (v2-v1);
  if (diff < 1.0e-5)
    return true;
  return false;
}

bool Same(const DataVector &data) {
  if (data.empty())
    return true;

  ValueType t = data[0]->target;
  for (size_t i = 1; i < data.size(); ++i) {
    if (!AlmostEqual(t, data[i]->target))
      return false;
  }
  return true;
}

ValueType Average(const DataVector & data) {
  if (data.empty())
    return 0;
  ValueType s = 0;
  ValueType c = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    s += data[i]->target * data[i]->weight;
    c += data[i]->weight;
  }
  return s/c;
}


bool FindSplit(DataVector *data, int *index, ValueType *value) {
  int n = gConf.number_of_feature;
  ValueType best_fitness = kValueTypeMax;

  for (int i = 0; i < n; ++i) {
    std::sort(data->begin(), data->end(), TupleCompare(i));
    int unknown = 0;
    ValueType s = 0;
    ValueType ss = 0;
    ValueType c = 0;

    while ((*data)[unknown]->feature[i] == kUnknownValue && unknown < n) {
      unknown++;
      s += (*data)[unknown]->target * (*data)[unknown]->weight;
      ss += Squared((*data)[unknown]->target) * (*data)[unknown]->weight;
      c += (*data)[unknown]->weight;
    }

    if (unknown == n) {
      continue;
    }

    ValueType fitness0 = ss - s*s/c;
    assert(fitness0 >= 0);

    s = 0;
    ss = 0;
    c = 0;
    for (int j = unknown; j < n; ++j) {
      s += (*data)[j]->target * (*data)[j]->weight;
      ss += Squared((*data)[j]->target) * (*data)[j]->weight;
      c += (*data)[j]->weight;
    }

    ValueType ls = 0, lss = 0, lc = 0;
    ValueType rs = s, rss = ss, rc = c;
    ValueType fitness1 = 0, fitness2 = 0;
    for (int j = unknown; j < n-1; ++j) {
      s = (*data)[j]->target * (*data)[j]->weight;
      ss = Squared((*data)[j]->target) * (*data)[j]->weight;
      c = (*data)[j]->weight;

      ls += s;
      lss += ss;
      lc += c;

      rs -= s;
      rss -= ss;
      rc -= c;

      ValueType f1 = (*data)[j]->feature[i];
      ValueType f2 = (*data)[j+1]->feature[i];
      if (AlmostEqual(f1, f2))
        continue;

      fitness1 = lss - ls*ls/lc;
      assert(fitness1 >= 0);
      fitness2 = rss - rs*rs/rc;
      assert(fitness2 >= 0);

      ValueType fitness = fitness0 + fitness1 + fitness2;
      if (best_fitness > fitness) {
        best_fitness = fitness;
        *index = i;
        *value = (f1+f2)/2;
      }
    }
  }

  return best_fitness != kValueTypeMax;
}

void SplitData(const DataVector &data, int index, ValueType value, DataVector *output) {
  for (size_t i = 0; i < data.size(); ++i) {
    if (data[i]->feature[index] == kUnknownValue) {
      output[Node::UNKNOWN].push_back(data[i]);
    } else if (data[i]->feature[index] < value) {
      output[Node::LT].push_back(data[i]);
    } else {
      output[Node::GE].push_back(data[i]);
    }
  }
}

}
