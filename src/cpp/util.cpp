// Author: qiyiping@gmail.com (Yiping Qi)

#include "util.hpp"

namespace gbdt {

size_t SplitString(const std::string& str,
                   const std::string& separator,
                   std::vector<std::string>* tokens) {
  tokens->clear();

  size_t start = 0;
  while (start < str.length()) {
    size_t end = str.find(separator, start);
    if (end == std::string::npos) {
      tokens->push_back(str.substr(start));
      break;
    } else {
      tokens->push_back(str.substr(start, end - start));
      start = end + separator.length();
    }
  }

  return tokens->size();
}

std::string JoinString(
    const std::vector<std::string>& parts,
    const std::string& separator) {
  if (parts.empty())
    return std::string();

  std::string result(parts[0]);
  std::vector<std::string>::const_iterator iter = parts.begin();
  ++iter;

  for (; iter != parts.end(); ++iter) {
    result += separator;
    result += *iter;
  }

  return result;
}

}
