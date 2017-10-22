#ifndef SCOPED_TIMERS_H
#define SCOPED_TIMERS_H
 
#include <sys/time.h>
 
class SystemTimer {
public:
  explicit SystemTimer(float& time)
    : time(time)
  {
    gettimeofday(&start, NULL);
  }
 
  ~SystemTimer()
  {
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) * 1e-6;
  }
private:
  struct timeval start;
  struct timeval stop;
  float& time;
};
 
#endif // SCOPED_TIMERS_H
