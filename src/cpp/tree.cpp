// Author: qiyiping@gmail.com (Yiping Qi)

#include "tree.hpp"
#include "math_util.hpp"
#include "util.hpp"
#include "loss.hpp"
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
void RegressionTree::Fit(DataVector *data,
                         size_t len,
                         Node *node,
                         size_t depth,
                         double *gain) {
  size_t max_depth = conf.max_depth;

  node->pred = conf.loss->GetRegionPrediction(*data, len);

  if (max_depth == depth
      || Same(*data, len)
      || len <= conf.min_leaf_size) {
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

  gain[node->index] += g;

  // increase feature cost if certain feature is used
  if (conf.enable_feature_tunning) {
    conf.feature_costs[node->index] += 1.0e-4;
  }

  node->child[Node::LT] = new Node();
  node->child[Node::GE] = new Node();

  Fit(&out[Node::LT], out[Node::LT].size(),
      node->child[Node::LT], depth+1, gain);
  Fit(&out[Node::GE], out[Node::GE].size(),
      node->child[Node::GE], depth+1, gain);

  if (!out[Node::UNKNOWN].empty()) {
    node->child[Node::UNKNOWN] = new Node();
    Fit(&out[Node::UNKNOWN], out[Node::UNKNOWN].size(),
        node->child[Node::UNKNOWN], depth+1, gain);
  }
}

ValueType RegressionTree::Predict(const Node *root, const Tuple &t) const {
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

ValueType RegressionTree::Predict(const Node *root, const Tuple &t, double *p) const {
  if (root->leaf) {
    return root->pred;
  }
  if (t.feature[root->index] == kUnknownValue) {
    if (root->child[Node::UNKNOWN]) {
      p[root->index] += (root->child[Node::UNKNOWN]->pred - root->pred);
      return Predict(root->child[Node::UNKNOWN], t, p);
    } else {
      return root->pred;
    }
  } else if (t.feature[root->index] < root->value) {
    p[root->index] += (root->child[Node::LT]->pred - root->pred);
    return Predict(root->child[Node::LT], t, p);
  } else {
    p[root->index] += (root->child[Node::GE]->pred - root->pred);
    return Predict(root->child[Node::GE], t, p);
  }
}

void RegressionTree::Fit(DataVector *data, size_t len) {
  assert(data->size() >= len);
  delete root;
  root = new Node();
  delete[] gain;
  gain = new double[conf.number_of_feature];
  for (size_t i = 0; i < conf.number_of_feature; ++i) {
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
    ns += std::to_string(nodes[i]->index);
    ns += " ";
    ns += std::to_string(nodes[i]->value);
    ns += " ";
    ns += std::to_string(nodes[i]->leaf);
    ns += " ";
    ns += std::to_string(nodes[i]->pred);
    for (int j = 0; j < Node::CHILDSIZE; ++j) {
      ns += " ";
      if (nodes[i]->child[j]) {
        size_t p = position_map[nodes[i]->child[j]];
        ns += std::to_string(p);
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
                             std::map<const void *, size_t> *position_map) const {
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
    n->index = std::stoi(items[0]);
    n->value = std::stod(items[1]);
    n->leaf = std::stoi(items[2]);
    n->pred = std::stod(items[3]);
    lt.push_back(std::stoi(items[4]));
    ge.push_back(std::stoi(items[5]));
    un.push_back(std::stoi(items[6]));

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


bool RegressionTree::FindSplit(DataVector *data, size_t m,
                               int *index, ValueType *value, double *gain) {
  size_t n = conf.number_of_feature;
  double best_fitness = std::numeric_limits<double>::max();

  std::vector<int> fv;
  for (int i = 0; i < n; ++i) {
    fv.push_back(i);
  }

  size_t fn = n;
  if (conf.feature_sample_ratio < 1) {
    fn = static_cast<size_t>(n*conf.feature_sample_ratio);
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

bool RegressionTree::GetImpurity(DataVector *data, size_t len,
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

    if (conf.enable_feature_tunning) {
      fitness *= conf.feature_costs[index];
    }

    if (*impurity > fitness) {
      *impurity = fitness;
      *value = (f1+f2)/2;
      *gain = fitness00 - fitness1 - fitness2;
    }
  }

  return *impurity != std::numeric_limits<double>::max();
}

void RegressionTree::SplitData(const DataVector &data, size_t len,
                               int index, ValueType value, DataVector *output) {
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


}
