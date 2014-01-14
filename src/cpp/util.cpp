// Author: qiyiping@gmail.com (Yiping Qi)

#include <util.hpp>

namespace gbdt {

size_t SplitString(const std::string& str,
                   const std::string& delimiters,
                   std::vector<std::string>* tokens) {
  tokens->clear();

  size_t start = str.find_first_not_of(delimiters);
  while (start != std::string::npos) {
    size_t end = str.find_first_of(delimiters, start + 1);
    if (end == std::string::npos) {
      tokens->push_back(str.substr(start));
      break;
    } else {
      tokens->push_back(str.substr(start, end - start));
      start = str.find_first_not_of(delimiters, end + 1);
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
    result += sep;
    result += *iter;
  }

  return result;
}

}
