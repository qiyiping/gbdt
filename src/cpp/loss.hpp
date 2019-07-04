#ifndef LOSS_H
#define LOSS_H

#include <map>
#include "data.hpp"

namespace gbdt {

///////////////////////////////////////////////////////////////////////
// Objective for function approximation
///////////////////////////////////////////////////////////////////////
class Objective {
 public:
  virtual double GetBias(DataVector &d, size_t len) const = 0;
  virtual double GetLoss(const Tuple& t, ValueType f) const = 0;
  virtual void UpdateGradient(Tuple* t, ValueType f) const = 0;
  virtual double GetRegionPrediction(DataVector &d, size_t len) const = 0;
  virtual std::string GetName() const = 0;
  virtual ~Objective() {}
};

class LossFactory {
 public:
  using CreateFn = Objective* (*) ();

  bool Register(const std::string &name, CreateFn creater);
  Objective* Create(const std::string &name);
  void GetAllCandidates(std::vector<std::string> *candidates);
  void PrintAllCandidates();
  bool LoadSharedLib(const std::string &path);
  ~LossFactory();
  static LossFactory* GetInstance();
 protected:
  LossFactory() {}
 private:
  std::map<std::string, CreateFn> creaters_;
  std::vector<void*> handles_;
};


#ifndef DECLARE_OBJECTIVE_REGISTRATION
#define DECLARE_OBJECTIVE_REGISTRATION(CLZ)                     \
  void register__ ## CLZ(void) __attribute__ ((constructor));
#endif

#ifndef DEFINE_OBJECTIVE_CREATOR
#define DEFINE_OBJECTIVE_CREATOR(CLZ)                           \
  static Objective* create__ ## CLZ() { return new CLZ(); }
#endif

#ifndef DEFINE_OBJECTIVE_REGISTRATION
#define DEFINE_OBJECTIVE_REGISTRATION(CLZ)                              \
  DEFINE_OBJECTIVE_CREATOR(CLZ)                                         \
  void register__ ## CLZ(void) {                                        \
    LossFactory::GetInstance()->Register(#CLZ, create__ ## CLZ);        \
  }
#endif

#ifndef SHARED_LIB_INIT_DECLARE
#define SHARED_LIB_INIT_DECLARE                 \
  extern "C" void init(LossFactory *factory);
#endif

#ifndef SHARED_LIB_INIT_BEGIN
#define SHARED_LIB_INIT_BEGIN                   \
  void init(LossFactory *factory) {
#endif

#ifndef SHARED_LIB_INIT_END
#define SHARED_LIB_INIT_END                     \
  }
#endif


#ifndef SHARED_LIB_INIT_REGISTER
#define SHARED_LIB_INIT_REGISTER(CLZ)           \
  factory->Register(#CLZ, create__ ## CLZ);
#endif

}  // gbdt

#endif /* LOSS_H */
