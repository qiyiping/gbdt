// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _CONFIG_H_
#define _CONFIG_H_

namespace gbdt {
class Configure {
 public:
  int number_of_feature;  // number of features
  int max_depth;          // max depth for each tree
  int iterations;         // number of trees in gbdt
  float shrinkage;        // shrinkage parameter
};

extern Configure gConf;
}

#endif /* _CONFIG_H_ */
