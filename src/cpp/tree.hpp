// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _TREE_H_
#define _TREE_H_
#include <map>
#include <vector>
#include "config.hpp"
#include "data.hpp"

namespace gbdt {
class Node {
 public:
  enum {LT, GE, UNKNOWN, CHILDSIZE};

  Node() {
    child[LT] = NULL;
    child[GE] = NULL;
    child[UNKNOWN] = NULL;
    index = -1;
    value = 0;
    leaf = false;
    pred = 0;
  }

  ~Node() {
    delete child[LT];
    delete child[GE];
    delete child[UNKNOWN];
  }

  static void Fit(DataVector *data,
                  Node *node,
                  int depth);

  static ValueType Predict(Node *root, const Tuple &t);

  Node *child[CHILDSIZE];
  int index;
  ValueType value;
  bool leaf;
  ValueType pred;
 private:
  DISALLOW_COPY_AND_ASSIGN(Node);
};

class RegressionTree {
 public:
  RegressionTree(const Configure &conf): root(NULL), gain(NULL), conf(conf) {}
  ~RegressionTree() {
    delete root;
    delete[] gain;
  }

  void Fit(DataVector *data) { Fit(data, data->size()); }
  void Fit(DataVector *data, size_t len);

  ValueType Predict(const Tuple &t) const;
  ValueType Predict(const Tuple &t, double *p) const;

  std::string Save() const;
  void Load(const std::string &s);

  double *GetGain() { return gain; }

 private:
  void Fit(DataVector *data,
           size_t len,
           Node *node,
           size_t depth,
           double *gain);

  ValueType Predict(const Node *node, const Tuple &t) const;
  ValueType Predict(const Node *node, const Tuple &t, double *p) const;

  void SaveAux(const Node *node,
               std::vector<const Node *> *nodes,
               std::map<const void *, size_t> *position_map) const;

 private:
  bool FindSplit(DataVector *data, size_t len,
                 int *index, ValueType *value, double *gain);
  bool GetImpurity(DataVector *data, size_t len,
                   int index, ValueType *value,
                   double *impurity, double *gain);

  static void SplitData(const DataVector &data, size_t len, int index, ValueType value, DataVector *output);

 private:
  Node *root;
  double *gain;
  Configure conf;

  DISALLOW_COPY_AND_ASSIGN(RegressionTree);
};

}

#endif /* _TREE_H_ */
