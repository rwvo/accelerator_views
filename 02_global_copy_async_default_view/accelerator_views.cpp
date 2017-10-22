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
    auto acc_view = acc.get_default_view();
    
    pinned_vector<double> host_data(size); // host pinned memory, alloc-ed with am_alloc, zero-initialized
    // auto device_ptr = hc::am_alloc(size * sizeof(int), acc, 0);
    // auto device_data = hc::array<int, 1>(extent<1>(size), acc_view, device_ptr);
    auto device_data = hc::array<double, 1>(extent<1>(size));
    // acc_view.copy_async(device_data.accelerator_pointer(), host_data.data(), size * sizeof(int));
    
    auto average = std::accumulate(host_data.begin(), host_data.end(), 0.0) / size;
    
    SHOW_TIME(hc::copy_async(host_data.data(), device_data));
    SHOW_TIME(busywork(device_data, 1, 0.0, 1.0));
    SHOW_TIME(hc::copy_async(device_data, host_data.data()));
    SHOW_TIME(acc_view.wait());
    average = std::accumulate(host_data.begin(), host_data.end(), 0.0) / size;
    std::cerr << "average: " << average << '\n';
    
    std::fill(host_data.begin(), host_data.end(), 0.0);
    SHOW_TIME(hc::copy_async(host_data.data(), device_data));
    SHOW_TIME(acc_view.create_marker());
    SHOW_TIME(busywork(device_data, 1, 0.0, 1.0));
    SHOW_TIME(acc_view.create_marker());
    SHOW_TIME(hc::copy_async(device_data, host_data.data()));
    SHOW_TIME(acc_view.wait());
    average = std::accumulate(host_data.begin(), host_data.end(), 0.0) / size;
    std::cerr << "average: " << average << '\n';
    
    std::fill(host_data.begin(), host_data.end(), 0.0);
    SHOW_TIME(hc::copy_async(host_data.data(), device_data));
    SHOW_TIME(acc_view.create_marker().wait());
    SHOW_TIME(busywork(device_data, 1, 0.0, 1.0));
    SHOW_TIME(acc_view.create_marker().wait());
    SHOW_TIME(hc::copy_async(device_data, host_data.data()));
    SHOW_TIME(acc_view.wait());
    average = std::accumulate(host_data.begin(), host_data.end(), 0.0) / size;
    std::cerr << "average: " << average << '\n';
    
    std::fill(host_data.begin(), host_data.end(), 0.0);
    SHOW_TIME(hc::copy(host_data.data(), device_data));
    SHOW_TIME(busywork(device_data, 1, 0.0, 1.0).wait());
    SHOW_TIME(hc::copy(device_data, host_data.data()));
    SHOW_TIME(acc_view.wait());
    average = std::accumulate(host_data.begin(), host_data.end(), 0.0) / size;
    std::cerr << "average: " << average << '\n';
    //am_free(device_ptr);
  }
  std::cerr << "total time: " << tm << " seconds\n";
}
