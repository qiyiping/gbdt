// Author: qiyiping@gmail.com (Yiping Qi)

namespace gbdt {
ValueType GBDT::Predict(const Tuple &t) const {
  if (!trees)
    return kUnknownValue;

  ValueType r = 0;
  for (int i = 0; i < gConf.iterations; ++i) {
    r += gConf.shrinkage * tree[i].Predict(t);
  }

  return r;
}

void GBDT::Fit(DataVector *d) {
  trees = new RegressionTree[gConf.iterations];
  for (int i = 0; i < gConf.iterations; ++i) {
    tree[i].Fit(d);

    
  }
}
}
