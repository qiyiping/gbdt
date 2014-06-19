#ifndef _GBDT_TIME_H_
#define _GBDT_TIME_H_
#include <time.h>
#include <string>
#include <stdint.h>

namespace gbdt {
time_t TimeTFromTimeString(const std::string &time_string,
                           const std::string &format = "%Y-%m-%d %H:%M:%S");
std::string TimeStringFromTimeT(time_t t,
                                const std::string &format = "%Y-%m-%d %H:%M:%S");

class Time;

class TimeDelta
{
 public:
  TimeDelta(int64_t us): delta_(us) {}

  static TimeDelta FromDays(int64_t days);
  static TimeDelta FromHours(int64_t hours);
  static TimeDelta FromMinutes(int64_t minutes);
  static TimeDelta FromSeconds(int64_t secs);
  static TimeDelta FromMilliseconds(int64_t ms);
  static TimeDelta FromMicroseconds(int64_t us);

  int64_t ToDays() const;
  int64_t ToHours() const;
  int64_t ToMinutes() const;
  int64_t ToSeconds() const;
  int64_t ToMilliseconds() const;
  int64_t ToMicroseconds() const;

  TimeDelta operator+(const TimeDelta &other) const { return TimeDelta(delta_ + other.delta_); }
  TimeDelta operator-(const TimeDelta &other) const { return TimeDelta(delta_ - other.delta_); }
  const TimeDelta& operator+=(const TimeDelta &other) { delta_ += other.delta_; return *this; }
  const TimeDelta& operator-=(const TimeDelta &other) { delta_ -= other.delta_; return *this; }
  template <typename T>
  TimeDelta operator*(const T &n) const { return TimeDelta(delta_ * n); }
  template <typename T>
  TimeDelta operator/(const T &n) const { return TimeDelta(delta_ / n); }
  template <typename T>
  const TimeDelta& operator*=(const T &n) { delta_ *= n; return *this; }
  template <typename T>
  const TimeDelta& operator/=(const T &n) { delta_ /= n; return *this; }

 private:
  friend class Time;

  int64_t delta_;
};

class Time
{
 public:
  static const int64_t kMillisecondsPerSecond = 1000;
  static const int64_t kMicrosecondsPerMillisecond = 1000;
  static const int64_t kMicrosecondsPerSecond = kMicrosecondsPerMillisecond *
                                                kMillisecondsPerSecond;
  static const int64_t kMicrosecondsPerMinute = kMicrosecondsPerSecond * 60;
  static const int64_t kMicrosecondsPerHour = kMicrosecondsPerMinute * 60;
  static const int64_t kMicrosecondsPerDay = kMicrosecondsPerHour * 24;
  static const int64_t kMicrosecondsPerWeek = kMicrosecondsPerDay * 7;
  static const int64_t kNanosecondsPerMicrosecond = 1000;
  static const int64_t kNanosecondsPerSecond = kNanosecondsPerMicrosecond *
                                               kMicrosecondsPerSecond;

  struct Exploded {
    int year;          // Four digit year "2007"
    int month;         // 1-based month (values 1 = January, etc.)
    int day_of_week;   // 0-based day of week (0 = Sunday, etc.)
    int day_of_month;  // 1-based day of month (1-31)
    int hour;          // Hour within the current day (0-23)
    int minute;        // Minute within the current hour (0-59)
    int second;        // Second within the current minute (0-59 plus leap
                       // seconds which may take it up to 60).
    int millisecond;   // Milliseconds within the current second (0-999)

    // A cursory test for whether the data members are within their
    // respective ranges. A 'true' return value does not guarantee the
    // Exploded value can be successfully converted to a Time value.
    bool HasValidValues() const;

    // Convert Exploded to struct tm
    void ToTM(struct tm *timestruct) const;
    // Convert struct tm to Exploded
    static Exploded FromTM(const struct tm &timestruct);
  };

  static Time Now();
  static Time FromTimeT(time_t t);
  static Time FromString(const std::string &timestring,
                         const std::string &format);
  static Time FromExploded(const Exploded &e);

  TimeDelta Diff(const Time &other) const { return TimeDelta(us_ - other.us_); }

  Time(int64_t us): us_(us) {}

  void Add(const TimeDelta &delta) { us_ += delta.delta_; }
  void Subtract(const TimeDelta &delta) { us_ -= delta.delta_; }

  time_t ToTimeT() const { return us_ / kMicrosecondsPerSecond; }

  std::string ToString(const std::string &format) const {
    return TimeStringFromTimeT(ToTimeT(), format);
  }

  Exploded ToExploded() const;

  Time operator+(const TimeDelta &delta) const { return Time(us_ + delta.delta_); }
  Time operator-(const TimeDelta &delta) const { return Time(us_ - delta.delta_); }
  const Time& operator+=(const TimeDelta &delta) { us_ += delta.delta_; return *this; }
  const Time& operator-=(const TimeDelta &delta) { us_ -= delta.delta_; return *this; }

  // Comparison operators
  bool operator==(Time other) const {
    return us_ == other.us_;
  }
  bool operator!=(Time other) const {
    return us_ != other.us_;
  }
  bool operator<(Time other) const {
    return us_ < other.us_;
  }
  bool operator<=(Time other) const {
    return us_ <= other.us_;
  }
  bool operator>(Time other) const {
    return us_ > other.us_;
  }
  bool operator>=(Time other) const {
    return us_ >= other.us_;
  }

 private:
  int64_t us_;
};

// static
inline TimeDelta TimeDelta::FromDays(int64_t days) {
  return TimeDelta(days * Time::kMicrosecondsPerDay);
}

// static
inline TimeDelta TimeDelta::FromHours(int64_t hours) {
  return TimeDelta(hours * Time::kMicrosecondsPerHour);
}

// static
inline TimeDelta TimeDelta::FromMinutes(int64_t minutes) {
  return TimeDelta(minutes * Time::kMicrosecondsPerMinute);
}

// static
inline TimeDelta TimeDelta::FromSeconds(int64_t secs) {
  return TimeDelta(secs * Time::kMicrosecondsPerSecond);
}

// static
inline TimeDelta TimeDelta::FromMilliseconds(int64_t ms) {
  return TimeDelta(ms * Time::kMicrosecondsPerMillisecond);
}

// static
inline TimeDelta TimeDelta::FromMicroseconds(int64_t us) {
  return TimeDelta(us);
}

inline int64_t TimeDelta::ToDays() const { return delta_ / Time::kMicrosecondsPerDay; }
inline int64_t TimeDelta::ToHours() const { return delta_ / Time::kMicrosecondsPerHour; }
inline int64_t TimeDelta::ToMinutes() const { return delta_ / Time::kMicrosecondsPerMinute; }
inline int64_t TimeDelta::ToSeconds() const { return delta_ / Time::kMicrosecondsPerSecond; }
inline int64_t TimeDelta::ToMilliseconds() const { return delta_ / Time::kMicrosecondsPerMillisecond; }
inline int64_t TimeDelta::ToMicroseconds() const { return delta_; }

inline Time Time::FromTimeT(time_t t) {
  return Time(t * kMicrosecondsPerSecond);
}

inline Time Time::FromString(const std::string &timestring,
                             const std::string &format) {
  return FromTimeT(TimeTFromTimeString(timestring, format));
}

class Elapsed {
 public:
  Elapsed():t_(Time::Now()) {}
  TimeDelta Tell() { return Time::Now().Diff(t_); }
  void Reset() { t_ = Time::Now(); }
 private:
  Time t_;
};

}


#endif /* _GBDT_TIME_H_ */
