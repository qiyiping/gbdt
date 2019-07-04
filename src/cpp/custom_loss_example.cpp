#include "custom_loss_example.hpp"

namespace gbdt {

double MyLoss::GetRegionPrediction(DataVector &d, size_t len) const {
  return .0;
}

DEFINE_OBJECTIVE_CREATOR(MyLoss)

SHARED_LIB_INIT_BEGIN
SHARED_LIB_INIT_REGISTER(MyLoss)
SHARED_LIB_INIT_END

}  // gbdt
