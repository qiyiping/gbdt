// Author: qiyiping@gmail.com (Yiping Qi)
#include <map>
#include <vector>
#include <string>
#include <exception>
#include <boost/lexical_cast.hpp>

namespace gbdt {
  class CmdOption {
   public:
    template<typename T>
    T Get(const std::string &option_key, const T &default_val) const {
      const auto r = options_.find(option_key);
      if (r == options_.end()) {
        return default_val;
      }
      try {
        return boost::lexical_cast<T>(r->second);
      } catch (const std::exception &e) {
        std::cerr << "option key: " << option_key
                  << ", option value: " << r->second
                  << ", exception message" << e.what()
                  << std::endl;
        throw;
      }
    }

    bool Contains(const std::string &option_key) const {
      const auto r = options_.find(option_key);
      return r != options_.end();
    }

    static CmdOption ParseOptions(int argc, char* argv[]) {
      CmdOption opt;
      int i = 1;
      while (i < argc) {
        std::string key;
        if (argv[i][0] == '-') {
          key = std::string(argv[i] + 1);
          i += 1;
          if (i < argc && argv[i][0] != '-') {
            opt.options_[key] = std::string(argv[i]);
            i += 1;
          } else {
            opt.options_[key] = std::string();
          }
        } else {
          opt.args_.push_back(argv[i]);
          i += 1;
        }
      }
      return opt;
    }
   protected:
    CmdOption() {}
   private:
    std::map<std::string, std::string> options_;
    std::vector<std::string> args_;
  };
}
