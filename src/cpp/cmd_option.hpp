// Author: qiyiping@gmail.com (Yiping Qi)
#include <map>
#include <string>
#include <iostream>

#include <stdlib.h>
#include <string.h>

namespace gbdt {

enum OptionType {
  STRING,
  DOUBLE,
  INT,
  BOOL
};



class CmdOption {
 public:
  union OptionValue {
    int int_val;
    bool bool_val;
    char string_val[100];
    double double_val;
  };
  struct OptionInfo {
    std::string name;
    std::string long_opt;
    std::string short_opt;
    OptionType type;
    bool is_set;
    bool is_required;
    OptionValue val;
  };

 public:
  bool AddOption(const std::string &long_opt,
                 const std::string &short_opt,
                 const std::string &name,
                 OptionType type,
                 bool is_required=false) {
    if (long2name_.find(long_opt) != long2name_.end()
        || short2name_.find(short_opt) != short2name_.end()
        || options_.find(name) != options_.end()) {
      return false;
    }
    long2name_[long_opt] = name;
    short2name_[short_opt] = name;
    OptionInfo info  = {
      name, long_opt, short_opt, type, false, is_required, {0}
    };
    options_[name] = info;
    return true;
  }

  bool AddOption(const std::string &long_opt,
                 const std::string &short_opt,
                 const std::string &name,
                 const std::string &default_val) {
    if (AddOption(long_opt, short_opt, name, OptionType::STRING)) {
      strncpy(options_[name].val.string_val, default_val.c_str(), 100);
      options_[name].is_set = true;
      return true;
    }
    return false;
  }

  bool AddOption(const std::string &long_opt,
                 const std::string &short_opt,
                 const std::string &name,
                 const char *default_val) {
    if (AddOption(long_opt, short_opt, name, OptionType::STRING)) {
      strncpy(options_[name].val.string_val, default_val, 100);
      options_[name].is_set = true;
      return true;
    }
    return false;
  }

  bool AddOption(const std::string &long_opt,
                 const std::string &short_opt,
                 const std::string &name,
                 double default_val) {
    if (AddOption(long_opt, short_opt, name, OptionType::DOUBLE)) {
      options_[name].val.double_val = default_val;
      options_[name].is_set = true;
      return true;
    }
    return false;
  }

  bool AddOption(const std::string &long_opt,
                 const std::string &short_opt,
                 const std::string &name,
                 int default_val) {
    if (AddOption(long_opt, short_opt, name, OptionType::INT)) {
      options_[name].val.int_val = default_val;
      options_[name].is_set = true;
      return true;
    }
    return false;
  }

  bool AddOption(const std::string &long_opt,
                 const std::string &short_opt,
                 const std::string &name,
                 bool default_val) {
    if (AddOption(long_opt, short_opt, name, OptionType::BOOL)) {
      options_[name].val.bool_val = default_val;
      options_[name].is_set = true;
      return true;
    }
    return false;
  }

  bool Get(const std::string &name, std::string *out) const {
    const auto r = options_.find(name);
    if (r != options_.end()
        || r->second.type == OptionType::STRING) {
      *out = r->second.val.string_val;
      return true;
    }
    return false;
  }

  bool Get(const std::string &name, double *out) const {
    const auto r = options_.find(name);
    if (r != options_.end()
        || r->second.type == OptionType::DOUBLE) {
      *out = r->second.val.double_val;
      return true;
    }
    return false;
  }

  bool Get(const std::string &name, int *out) const {
    const auto r = options_.find(name);
    if (r != options_.end()
        || r->second.type == OptionType::INT) {
      *out = r->second.val.int_val;
      return true;
    }
    return false;
  }

  bool Get(const std::string &name, bool *out) const {
    const auto r = options_.find(name);
    if (r != options_.end()
        || r->second.type == OptionType::BOOL) {
      *out = r->second.val.bool_val;
      return true;
    }
    return false;
  }

  bool ParseOptions(int argc, char* argv[]) {
    int i = 1;
    while (i < argc) {
      std::string t(argv[i]);
      OptionInfo *info = NULL;
      if (t.substr(0,2) == "--") {
        info = GetByLongName(t.substr(2));
      } else if (t.substr(0,1) == "-") {
        info = GetByShortName(t.substr(1));
      }

      i++;

      if (info && i < argc) {
        switch (info->type) {
          case OptionType::STRING:
            strcpy(info->val.string_val, argv[i]);
            info->is_set = true;
            break;
          case OptionType::INT:
            info->val.int_val = std::stoi(argv[i]);
            info->is_set = true;
            break;
          case OptionType::DOUBLE:
            info->val.double_val = std::stod(argv[i]);
            info->is_set = true;
            break;
          case OptionType::BOOL:
            info->val.bool_val = (strcmp(argv[i], "true") == 0);
            info->is_set = true;
            break;
          default:
            break;
        }
        i++;
      }
    }

    return IsValid();
  }

  bool IsValid() {
    auto iter = options_.begin();
    for (; iter != options_.end(); ++iter) {
      if (iter->second.is_required && (!iter->second.is_set)) {
        return false;
      }
    }
    return true;
  }

  void Help() {
    auto iter = options_.begin();
    std::cout << "OPTIONS:" << std::endl;
    for (; iter != options_.end(); ++iter) {
      std::cout << "\t" << "-" << iter->second.short_opt << ", --" << iter->second.long_opt << "  <value>" << std::endl
                << "\t\t" << (iter->second.is_required? "Required argument,": "Optional argument,") << "current value: ";
      if (iter->second.is_set) {
        switch (iter->second.type) {
          case OptionType::STRING:
            std::cout << iter->second.val.string_val;
            break;
          case OptionType::INT:
            std::cout << iter->second.val.int_val;
            break;
          case OptionType::DOUBLE:
            std::cout << iter->second.val.double_val;
            break;
          case OptionType::BOOL:
            std::cout << (iter->second.val.bool_val? "true": "false");
            break;
          default:
            break;
        }
      } else {
        std::cout << "N/A";
      }

      std::cout << std::endl;
    }
  }

  CmdOption() {}

 private:
  OptionInfo *GetByLongName(const std::string &long_opt) {
    auto r = long2name_.find(long_opt);
    if (r == long2name_.end()) {
      return NULL;
    }
    return &options_[r->second];
  }


  OptionInfo *GetByShortName(const std::string &short_opt) {
    auto r = short2name_.find(short_opt);
    if (r == short2name_.end()) {
      return NULL;
    }
    return &options_[r->second];
  }

 private:
  std::map<std::string, OptionInfo> options_;
  std::map<std::string, std::string> long2name_;
  std::map<std::string, std::string> short2name_;
};
}
