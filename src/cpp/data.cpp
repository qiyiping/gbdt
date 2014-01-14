#include "data.hpp"
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <iostream>


namespace gbdt {
static const std::string kItemDelimiter = " ";
static const std::string kKVDelimiter = ":";
Configure gConf;

std::string Tuple::ToString() const {
  if (feature == NULL)
    return std::string();

  std::string result = boost::lexical_cast<std::string>(label);
  result += kItemDelimiter;
  result += boost::lexical_cast<std::string>(weight);

  int n = gConf.number_of_feature;
  for (int i = 0; i < n; i++) {
    if (feature[i] == kUnknownValue)
      continue;
    result += kItemDelimiter;
    result += boost::lexical_cast<std::string>(i);
    result += kKVDelimiter;
    result += boost::lexical_cast<std::string>(feature[i]);
  }

  return result;
}

Tuple* Tuple::FromString(const std::string &l) {
  Tuple* result = new Tuple();
  int n = gConf.number_of_feature;
  result->feature = new ValueType[n];
  for (int i = 0; i < n; ++i) {
    result->feature[i] = kUnknownValue;
  }

  std::vector<std::string> tokens;
  if (SplitString(l, kItemDelimiter, &tokens) < 2) {
    delete result;
    return NULL;
  }

  result->label = boost::lexical_cast<ValueType>(tokens[0]);
  result->weight = boost::lexical_cast<ValueType>(tokens[1]);

  result->target = result->label;

  for (size_t i = 2; i < tokens.size(); ++i) {
    size_t found = tokens[i].find(kKVDelimiter);
    if (found == std::string::npos) {
      std::cerr << "feature value pair with wrong format: " << tokens[i];
      continue;
    }
    int index = boost::lexical_cast<int>(tokens[i].substr(0, found));
    if (index < 0 || index >= n) {
      std::cerr << "feature index out of boundary: " << index;
      continue;
    }
    ValueType value = boost::lexical_cast<ValueType>(tokens[i].substr(found+1));
    result->feature[index] = value;
  }

  return result;
}

void CleanDataVector(DataVector *data) {
  DataVector::iterator iter = data->begin();
  for (; iter != data->end(); ++iter) {
    delete *iter;
  }
}

bool LoadDataFromFile(const std::string &path, DataVector *data) {
  data->clear();
  std::ifstream stream(path.c_str());
  if (!stream) {
    return false;
  }

  std::string l;
  while(std::getline(stream, l)) {
    data->push_back(Tuple::FromString(l));
  }

  return true;
}

}
