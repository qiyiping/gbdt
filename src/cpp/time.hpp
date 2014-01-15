// Author: qiyiping@gmail.com (Yiping Qi)

#ifndef _TIME_H_
#define _TIME_H_
#include <sys/time.h>

namespace gbdt {

inline long CurrentTimeInMilliSeconds() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec*1000+tv.tv_usec/1000);
}

class Elapsed {
 public:
  Elapsed() { start = CurrentTimeInMilliSeconds(); }
  long Tell() { return CurrentTimeInMilliSeconds() - start; }
  void Reset() { start = CurrentTimeInMilliSeconds(); }
 private:
  long start;
};

}

#endif /* _TIME_H_ */
