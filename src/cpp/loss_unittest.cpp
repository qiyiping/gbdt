#include "loss.hpp"
#include "common_loss.hpp"
#include <iostream>

#include "cmd_option.hpp"
using namespace gbdt;

int main(int argc, char **argv)
{
  CmdOption opt;
  opt.AddOption("custom_loss_so", "f", "custom_loss_so", "");
  opt.ParseOptions(argc, argv);

  std::string custom_loss_so;
  opt.Get("custom_loss_so", &custom_loss_so);

  LossFactory::GetInstance()->LoadSharedLib(custom_loss_so);
  LossFactory::GetInstance()->PrintAllCandidates();
  return 0;
}
