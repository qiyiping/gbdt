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
  float feature_sample_ratio; // portion of features to be splited
  float data_sample_ratio;// portion of data to be fitted in each iteration

  Configure(): feature_sample_ratio(1), data_sample_ratio(1) {}
};

extern Configure gConf;
}

#endif /* _CONFIG_H_ */
