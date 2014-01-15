// Author: qiyiping@gmail.com (Yiping Qi)

#include "tree.hpp"
#include "fitness.hpp"

#include <iostream>

namespace gbdt {
void RegressionTree::Fit(DataVector *data,
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

  // std::cout << "depth: " << depth << " index: " << node->index << " value: " << node->value << std::endl;

  DataVector out[Node::CHILDSIZE];

  SplitData(*data, node->index, node->value, out);
  if (out[Node::LT].empty() || out[Node::GE].empty()) {
    node->leaf = true;
    node->pred = Average(*data);
    return;
  }

  node->child[Node::LT] = new Node();
  node->child[Node::GE] = new Node();

  Fit(&out[Node::LT], node->child[Node::LT], depth+1);
  Fit(&out[Node::GE], node->child[Node::GE], depth+1);

  if (!out[Node::UNKNOWN].empty()) {
    node->child[Node::UNKNOWN] = new Node();
    Fit(&out[Node::UNKNOWN], node->child[Node::UNKNOWN], depth+1);
  }
}

ValueType RegressionTree::Predict(const Node *root, const Tuple &t) {
  if (root->leaf) {
    return root->pred;
  }
  if (t.feature[root->index] == kUnknownValue) {
    return Predict(root->child[Node::UNKNOWN], t);
  } else if (t.feature[root->index] < root->value) {
    return Predict(root->child[Node::LT], t);
  } else {
    return Predict(root->child[Node::GE], t);
  }
}

void RegressionTree::Fit(DataVector *data) {
  delete root;
  root = new Node();
  Fit(data, root, 0);
}

ValueType RegressionTree::Predict(const Tuple &t) const {
  return Predict(root, t);
}

}
