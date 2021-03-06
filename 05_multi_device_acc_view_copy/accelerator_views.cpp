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

#define SHOW_TIME(fun_call)						\
  {									\
    float tm;								\
    {									\
      SystemTimer timer(tm);						\
      std::wcerr << #fun_call << ": ";					\
      fun_call;								\
    }									\
    auto GiB = 1.0 * size * sizeof(double) / (1024 * 1024 * 1024);	\
    std::wcerr << tm << " seconds, " << GiB/tm << "GiB/s\n"; \
  }

void show_accelerators(const std::vector<hc::accelerator>& accelerators){
  std::wcerr << "CPU/GPU devices:\n\n";
  for(const auto& acc: accelerators){
    std::wcerr << acc.get_device_path() << '\n';
    std::wcerr << acc.get_description() << '\n';
    std::wcerr << "device memory " << acc.get_dedicated_memory() << '\n';
    std::wcerr << "has display: " << acc.get_has_display() << "\n\n";
  }
}

std::vector<int> get_devices(const std::vector<hc::accelerator>& accelerators){
  std::vector<int> devices;
  for(std::size_t i=0; i!= accelerators.size(); ++i){
    if(accelerators[i].get_device_path() != L"cpu"){
      devices.push_back(i);
    }
  }
  return devices;
}


int main(){
  float tm;
  {
    SystemTimer timer(tm);
    constexpr size_t size = 1_GiB;
    hc::accelerator default_acc;
    auto accelerators = accelerator::get_all();
    auto devices = get_devices(accelerators);
    std::wcerr << "number of GPU devices: " << devices.size() << "\n\n";
    show_accelerators(accelerators);
    if(devices.size() == 0){
      std::wcerr << "No GPU devices found, exiting.\n";
    }

    auto devno1 = devices.front();
    auto devno2 = devices.back(); // if there's only one GPU, front and back are the same device.

    auto acc1 = accelerators[devno1];
    auto acc2 = accelerators[devno2];

    auto acc_view1 = acc1.create_view();
    auto acc_view2 = acc2.create_view();
    
    pinned_vector<double> host_data1(size, 3.1415927); // host pinned memory, alloc-ed with am_alloc, all values initialized to Pi
    pinned_vector<double> host_data2(size); 
    auto device_ptr1 = hc::am_alloc(size * sizeof(double), acc1, 0);
    auto device_ptr2 = hc::am_alloc(size * sizeof(double), acc2, 0);
    auto device_data1 = hc::array<double, 1>(extent<1>(size), acc_view1, device_ptr1);
    auto device_data2 = hc::array<double, 1>(extent<1>(size), acc_view2, device_ptr2);

    bool mapping_succeeded = (hc::am_map_to_peers(device_ptr2, 1, &acc1) == AM_SUCCESS);
    std::wcerr << "Mapping succeeded: " << mapping_succeeded << '\n';

    std::wcerr << "Copying from host -> device " << devno1 << " -> device " << devno2 << " -> host:\n";
    
    SHOW_TIME(acc_view1.copy_async(host_data1.data(), device_data1.accelerator_pointer(), size * sizeof(double)).wait());
    SHOW_TIME(acc_view1.copy_async(device_data1.accelerator_pointer(), device_data2.accelerator_pointer(),
                                   size * sizeof(double)).wait());
    SHOW_TIME(acc_view2.copy_async(device_data2.accelerator_pointer(), host_data2.data(), size * sizeof(double)).wait());
    
    auto average2 = std::accumulate(host_data2.begin(), host_data2.end(), 0.0) / size;
    std::wcerr << "average2: " << average2 << '\n'; // expected value: 3.1415927
    
    am_free(device_ptr1);
    am_free(device_ptr2);
  }
  std::wcerr << "total time: " << tm << " seconds\n";
}
