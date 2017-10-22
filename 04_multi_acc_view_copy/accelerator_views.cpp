#include <hc.hpp>
#include <hc_am.hpp>
#include <pinned_vector.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "scoped_timers.hpp"

using namespace hc;

constexpr size_t operator"" _MiB(unsigned long long MiB){ return MiB * 1024 * 1024 / sizeof(double); }
constexpr size_t operator"" _GiB(unsigned long long GiB){ return GiB * 1024 * 1024 * 1024 / sizeof(double); }

// recursive template for doing n floating point operations
template<int n> double flops(double arg) [[hc]] { return arg + arg * flops<n-2>(arg); }
template<> double flops<1>(double arg) [[hc]] { return arg + arg; }
template<> double flops<0>(double arg) [[hc]] { return arg; }


hc::completion_future busywork(hc::array<double,1>& device_data, std::size_t reps, double threshold, double final_value)
{
  return parallel_for_each(device_data.get_accelerator_view(), device_data.get_extent(),
			   [=,&device_data](hc::index<1> idx)[[hc]]
			   {
			     for(std::size_t rep = 0; rep != reps; ++rep){
			       device_data[idx] = flops<1000>(device_data[idx]);
			     }
			     if(device_data[idx] >= threshold){
			       device_data[idx] = final_value;
			     }
			   });
}

#define SHOW_TIME(fun_call) \
  {\
    float tm;\
    {\
      SystemTimer timer(tm);\
      fun_call;\
    }\
    std::cerr << #fun_call << ": " << tm << " seconds\n";\
  }

int main(){
  float tm;
  {
    SystemTimer timer(tm);
    constexpr size_t size = 1_GiB;
    hc::accelerator acc;
    auto acc_view1 = acc.create_view();
    auto acc_view2 = acc.create_view();
    
    pinned_vector<double> host_data1(size); // host pinned memory, alloc-ed with am_alloc, zero-initialized
    pinned_vector<double> host_data2(size); 
    auto device_ptr1 = hc::am_alloc(size * sizeof(double), acc, 0);
    auto device_ptr2 = hc::am_alloc(size * sizeof(double), acc, 0);
    auto device_data1 = hc::array<double, 1>(extent<1>(size), acc_view1, device_ptr1);
    auto device_data2 = hc::array<double, 1>(extent<1>(size), acc_view2, device_ptr2);
    
    SHOW_TIME(acc_view1.copy_async(host_data1.data(), device_data1.accelerator_pointer(), size * sizeof(double)));
    SHOW_TIME(busywork(device_data1, 1, 0.0, 1.0));
    SHOW_TIME(acc_view1.copy_async(device_data1.accelerator_pointer(), host_data1.data(), size * sizeof(double)));

    // uncomment to see what the total time is for doing the work on both accelerator_views
    // sequentially.
    // acc_view1.wait();
    
    SHOW_TIME(acc_view2.copy_async(host_data2.data(), device_data2.accelerator_pointer(), size * sizeof(double)));
    SHOW_TIME(busywork(device_data2, 1, 0.0, 1.0));
    SHOW_TIME(acc_view2.copy_async(device_data2.accelerator_pointer(), host_data2.data(), size * sizeof(double)));
    
    SHOW_TIME(acc_view1.wait());
    auto average1 = std::accumulate(host_data1.begin(), host_data1.end(), 0.0) / size;
    std::cerr << "average1: " << average1 << '\n';

    SHOW_TIME(acc_view2.wait());
    auto average2 = std::accumulate(host_data2.begin(), host_data2.end(), 0.0) / size;
    std::cerr << "average2: " << average2 << '\n';

    
    am_free(device_ptr1);
    am_free(device_ptr2);
  }
  std::cerr << "total time: " << tm << " seconds\n";
}
