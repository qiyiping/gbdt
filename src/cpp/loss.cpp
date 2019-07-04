#include "loss.hpp"
#include <dlfcn.h>
#include <iostream>

namespace gbdt {

bool LossFactory::Register(const std::string &name, CreateFn creater) {
  auto r = creaters_.insert(std::make_pair(name, creater));
  return r.second;
}

Objective* LossFactory::Create(const std::string &name) {
  auto iter = creaters_.find(name);
  if (iter == creaters_.end()) {
    return NULL;
  }
  return iter->second();
}

void LossFactory::GetAllCandidates(std::vector<std::string> *candidates) {
  candidates->clear();
  for (auto iter = creaters_.begin(); iter != creaters_.end(); ++iter) {
    candidates->push_back(iter->first);
  }
}

void LossFactory::PrintAllCandidates() {
  std::cout << "Objective Candidates: ";
  for (auto iter = creaters_.begin(); iter != creaters_.end(); ++iter) {
    std::cout << iter->first << ",";
  }
  std::cout << std::endl;
}

LossFactory* LossFactory::GetInstance() {
  static LossFactory inst;
  return &inst;
}


bool LossFactory::LoadSharedLib(const std::string &path) {
  if (path.empty()) {
    return false;
  }
  void *handle = dlopen(path.c_str(), RTLD_LAZY);
  if (!handle) {
    std::cerr << dlerror() << std::endl;
    return false;
  }

  using InitFn = void (*)(LossFactory *);
  InitFn init_fn = (InitFn) dlsym(handle, "init");
  if (!init_fn) {
    std::cerr << dlerror() << std::endl;
    return false;
  }
  init_fn(this);

  handles_.push_back(handle);
  return true;
}

LossFactory::~LossFactory() {
  for (auto iter = handles_.begin();
       iter != handles_.end();
       ++iter) {
    if (dlclose(*iter) < 0) {
      std::cerr << dlerror() << std::endl;
    }
  }
}

}  // gbdt
