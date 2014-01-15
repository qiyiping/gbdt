// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _TREE_H_
#define _TREE_H_

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
  RegressionTree(): root(NULL) {}
  ~RegressionTree() { delete root; }

  void Fit(DataVector *data);
  ValueType Predict(const Tuple &t) const;

 private:
  static void Fit(DataVector *data,
                  Node *node,
                  int depth);

  static ValueType Predict(const Node *node, const Tuple &t);

 private:
  Node *root;

  DISALLOW_COPY_AND_ASSIGN(RegressionTree);
};

}

#endif /* _TREE_H_ */
