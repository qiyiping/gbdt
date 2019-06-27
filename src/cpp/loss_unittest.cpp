#include "loss.hpp"
#include <iostream>
using namespace gbdt;

int main(int argc, char **argv)
{
  std::vector<std::string> candidates;
  LossFactory::GetInstance()->GetAllCandidates(&candidates);
  for (auto iter = candidates.begin(); iter != candidates.end(); ++iter) {
    std::cout << *iter << std::endl;
  }
  return 0;
}
