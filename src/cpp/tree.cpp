// Author: qiyiping@gmail.com (Yiping Qi)

#include "tree.hpp"
#include "fitness.hpp"

namespace gbdt {
void Node::Fit(DataVector *data,
               Node *node,
               int depth) {
  int max_depth = gConf.max_depth;

  if (max_depth == depth || Same(*data)) {
    node->leaf = true;
    node->pred = Average(*data);
    return;
  }

  if (!FindSplit(data, &(node->index), &(node->value))) {
    node->leaf = true;
    node->pred = Average(*data);
    return;
  }

  DataVector out[CHILDSIZE];

  SplitData(data, node->index, node->value, out);
  if (out[LT].empty() || out[GE].empty()) {
    node->leaf = true;
    node->pred = Average(*data);
    return;
  }

  node->child[LT] = new Node();
  node->child[GE] = new Node();

  Fit(&out[LT], node->child[LT], depth+1);
  Fit(&out[GE], node->child[GE], depth+1);

  if (!out[UNKNOWN].empty()) {
    node->child[UNKNOWN] = new Node();
    Fit(&out[UNKNOWN], node->child[UNKNOWN], depth+1);
  }
}

ValueType Node::Predict(Node *root, const Tuple &t) {
  if (root->leaf) {
    return root->pred;
  }
  if (t.feature[root->index] == kUnknownValue) {
    return Predict(root->child[UNKNOWN], t);
  } else if (t.feature[root->index] < root->value) {
    return Predict(root->child[LT], t);
  } else {
    return Predict(root->child[GE], t);
  }
}

}
