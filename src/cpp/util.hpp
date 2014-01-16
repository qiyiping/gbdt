// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _UTIL_H_
#define _UTIL_H_

#include <string>
#include <vector>

namespace gbdt {

// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
#define DISALLOW_COPY_AND_ASSIGN(TypeName)      \
  TypeName(const TypeName&);                    \
  void operator=(const TypeName&)

// A macro to suppress the compiler's warning about unused variable
#define UNUSED(expr) do { (void)(expr); } while (0)


std::string JoinString(
    const std::vector<std::string>& parts,
    const std::string& separator);

size_t SplitString(const std::string& str,
                   const std::string& separator,
                   std::vector<std::string>* tokens);

template <typename T>
void FreeVector(std::vector<T> *v) {
  std::vector<T> t;
  v->swap(t);
}

}

#endif /* _UTIL_H_ */
