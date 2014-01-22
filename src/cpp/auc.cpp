// Author: qiyiping@gmail.com (Yiping Qi)

#include "auc.hpp"
#include <iostream>

namespace gbdt {
void Auc::Add(ValueType score, ValueType label) {
  if (label > 0) {
    positive_scores.push_back(score);

    if (score >= threshold) {
      confusion_table[3] += 1;
    } else {
      confusion_table[2] += 1;
    }
  } else {
    negative_scores.push_back(score);
    if (score < threshold) {
      confusion_table[0] += 1;
    } else {
      confusion_table[1] += 1;
    }
  }
}

ValueType Auc::CalculateAuc() {
  std::sort(negative_scores.begin(), negative_scores.end());
  std::sort(positive_scores.begin(), positive_scores.end());

  int n0 = negative_scores.size();
  int n1 = positive_scores.size();

  if (n0 == 0 || n1 == 0) {
    return 0.5;
  }

  // scan the data
  int i0 = 0;
  int i1 = 0;
  double rank = 1;
  double rankSum = 0;
  while (i0 < n0 && i1 < n1) {

    double v0 = negative_scores[i0];
    double v1 = positive_scores[i1];

    if (v0 < v1) {
      i0++;
      rank++;
    } else if (v1 < v0) {
      i1++;
      rankSum += rank;
      rank++;
    } else {
      // ties have to be handled delicately
      double tieScore = v0;

      // how many negatives are tied?
      int k0 = 0;
      while (i0 < n0 && negative_scores[i0] == tieScore) {
        k0++;
        i0++;
      }

      // and how many positives
      int k1 = 0;
      while (i1 < n1 && positive_scores[i1] == tieScore) {
        k1++;
        i1++;
      }

      // we found k0 + k1 tied values which have
      // ranks in the half open interval [rank, rank + k0 + k1)
      // the average rank is assigned to all
      rankSum += (rank + (k0 + k1 - 1) / 2.0) * k1;
      rank += k0 + k1;
    }
  }

  if (i1 < n1) {
    rankSum += (rank + (n1 - i1 - 1) / 2.0) * (n1 - i1);
    rank += (n1 - i1);
  }

  return (rankSum / n1 - (n1 + 1) / 2) / n0;
}

void Auc::PrintConfusionTable() const {
  std::cout << confusion_table[0] << "\t" << confusion_table[1] << std::endl
            << confusion_table[2] << "\t" << confusion_table[3] << std::endl;
}
}
