// Author: qiyiping@gmail.com (Yiping Qi)

#include "tree.hpp"
#include "fitness.hpp"
#include "util.hpp"
#include <boost/lexical_cast.hpp>
#include <cassert>

namespace gbdt {
void RegressionTree::Fit(DataVector *data,
                         size_t len,
                         Node *node,
                         size_t depth,
                         double *gain) {
  size_t max_depth = g_conf.max_depth;

  if (g_conf.loss == SQUARED_ERROR) {
    node->pred = Average(*data, len);
  } else if (g_conf.loss == LOG_LIKELIHOOD) {
    node->pred = LogitOptimalValue(*data, len);
  }

  if (max_depth == depth
      || Same(*data, len)
      || len <= g_conf.min_leaf_size) {
    node->leaf = true;
    return;
  }

  double g = 0.0;
  if (!FindSplit(data, len, &(node->index), &(node->value), &g)) {
    node->leaf = true;
    return;
  }

  DataVector out[Node::CHILDSIZE];

  SplitData(*data, len, node->index, node->value, out);
  if (out[Node::LT].empty() || out[Node::GE].empty()) {
    node->leaf = true;
    return;
  }

  // update gain
  if (gain[node->index] < g) {
    gain[node->index] = g;
  }

  // increase feature cost if certain feature is used
  if (g_conf.feature_costs && g_conf.enable_feature_tunning) {
    g_conf.feature_costs[node->index] += 1.0e-4;
  }

  node->child[Node::LT] = new Node();
  node->child[Node::GE] = new Node();

  Fit(&out[Node::LT], node->child[Node::LT], depth+1, gain);
  Fit(&out[Node::GE], node->child[Node::GE], depth+1, gain);

  if (!out[Node::UNKNOWN].empty()) {
    node->child[Node::UNKNOWN] = new Node();
    Fit(&out[Node::UNKNOWN], node->child[Node::UNKNOWN], depth+1, gain);
  }
}

ValueType RegressionTree::Predict(const Node *root, const Tuple &t) {
  if (root->leaf) {
    return root->pred;
  }
  if (t.feature[root->index] == kUnknownValue) {
    if (root->child[Node::UNKNOWN]) {
      return Predict(root->child[Node::UNKNOWN], t);
    } else {
      return root->pred;
    }
  } else if (t.feature[root->index] < root->value) {
    return Predict(root->child[Node::LT], t);
  } else {
    return Predict(root->child[Node::GE], t);
  }
}

ValueType RegressionTree::Predict(const Node *root, const Tuple &t, double *p) {
  if (root->leaf) {
    return root->pred;
  }
  if (t.feature[root->index] == kUnknownValue) {
    if (root->child[Node::UNKNOWN]) {
      p[root->index] += (root->child[Node::UNKNOWN]->pred - root->pred);
      return Predict(root->child[Node::UNKNOWN], t);
    } else {
      return root->pred;
    }
  } else if (t.feature[root->index] < root->value) {
    p[root->index] += (root->child[Node::LT]->pred - root->pred);
    return Predict(root->child[Node::LT], t);
  } else {
    p[root->index] += (root->child[Node::GE]->pred - root->pred);
    return Predict(root->child[Node::GE], t);
  }
}

void RegressionTree::Fit(DataVector *data, size_t len) {
  assert(data->size() >= len);
  delete root;
  root = new Node();
  delete[] gain;
  gain = new double[g_conf.number_of_feature];
  for (size_t i = 0; i < g_conf.number_of_feature; ++i) {
    gain[i] = 0.0;
  }
  Fit(data, len, root, 0, gain);
}

ValueType RegressionTree::Predict(const Tuple &t) const {
  return Predict(root, t);
}

ValueType RegressionTree::Predict(const Tuple &t, double *p) const {
  return Predict(root, t, p);
}

std::string RegressionTree::Save() const {
  std::vector<const Node *> nodes;
  std::map<const void *, size_t> position_map;
  SaveAux(root, &nodes, &position_map);

  if (nodes.empty()) return std::string();

  std::vector<std::string> vs;
  for (size_t i = 0; i < nodes.size(); ++i) {
    std::string ns;
    ns += boost::lexical_cast<std::string>(nodes[i]->index);
    ns += " ";
    ns += boost::lexical_cast<std::string>(nodes[i]->value);
    ns += " ";
    ns += boost::lexical_cast<std::string>(nodes[i]->leaf);
    ns += " ";
    ns += boost::lexical_cast<std::string>(nodes[i]->pred);
    for (int j = 0; j < Node::CHILDSIZE; ++j) {
      ns += " ";
      if (nodes[i]->child[j]) {
        size_t p = position_map[nodes[i]->child[j]];
        ns += boost::lexical_cast<std::string>(p);
      } else {
        ns += "0";
      }
    }
    vs.push_back(ns);
  }

  return JoinString(vs, "\n");
}

void RegressionTree::SaveAux(const Node *node,
                             std::vector<const Node *> *nodes,
                             std::map<const void *, size_t> *position_map) {
  if (!node) return;
  nodes->push_back(node);
  position_map->insert(std::make_pair<const void *, size_t>(node, nodes->size() -1));

  SaveAux(node->child[Node::LT], nodes, position_map);
  SaveAux(node->child[Node::GE], nodes, position_map);
  SaveAux(node->child[Node::UNKNOWN], nodes, position_map);
}

void RegressionTree::Load(const std::string &s) {
  delete root;
  std::vector<std::string> vs;
  SplitString(s, "\n", &vs);

  std::vector<Node *> nodes;
  std::vector<std::string> items;

  std::vector<int> lt;
  std::vector<int> ge;
  std::vector<int> un;
  for (size_t i = 0; i < vs.size(); ++i) {
    Node *n = new Node();
    SplitString(vs[i], " ", &items);
    n->index = boost::lexical_cast<int>(items[0]);
    n->value = boost::lexical_cast<ValueType>(items[1]);
    n->leaf = boost::lexical_cast<bool>(items[2]);
    n->pred = boost::lexical_cast<ValueType>(items[3]);
    lt.push_back(boost::lexical_cast<int>(items[4]));
    ge.push_back(boost::lexical_cast<int>(items[5]));
    un.push_back(boost::lexical_cast<int>(items[6]));

    nodes.push_back(n);
  }

  for (size_t i = 0; i < nodes.size(); ++i) {
    if (lt[i] > 0) {
      nodes[i]->child[Node::LT] = nodes[lt[i]];
    }
    if (ge[i] > 0) {
      nodes[i]->child[Node::GE] = nodes[ge[i]];
    }
    if (un[i] > 0) {
      nodes[i]->child[Node::UNKNOWN] = nodes[un[i]];
    }
  }

  root = nodes[0];
}

}
