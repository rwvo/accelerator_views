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
    auto devno2 = devices.back();

    auto acc0 = accelerators[0];
    auto acc1 = accelerators[devno1];
    auto acc2 = accelerators[devno2];
    auto acc_view0 = acc0.create_view();
    auto acc_view1 = acc1.create_view();
    auto acc_view2 = acc2.create_view();
    
    pinned_vector<double> host_data1(size, 3.1415927); // host pinned memory, alloc-ed with am_alloc, zero-initialized
    pinned_vector<double> host_data2(size); 
    auto device_ptr1 = hc::am_alloc(size * sizeof(double), acc1, 0);
    auto device_ptr2 = hc::am_alloc(size * sizeof(double), acc2, 0);
    auto device_data1 = hc::array<double, 1>(extent<1>(size), acc_view1, device_ptr1);
    auto device_data2 = hc::array<double, 1>(extent<1>(size), acc_view2, device_ptr2);

    std::wcerr << "device_ptr1: " << (void*)device_ptr1 << '\n';
    std::wcerr << "device_ptr2: " << (void*)device_ptr2 << '\n';

    hc::AmPointerInfo devPtrInfo1(NULL, NULL, NULL, 0, default_acc, 0, 0);
    hc::AmPointerInfo devPtrInfo2(NULL, NULL, NULL, 0, default_acc, 0, 0);
    bool inTracker1 = hc::am_memtracker_getinfo(&devPtrInfo1, device_ptr1) == AM_SUCCESS;
    bool inTracker2 = hc::am_memtracker_getinfo(&devPtrInfo2, device_ptr2) == AM_SUCCESS;

    std::wcerr << "inTracker1: " << inTracker1 << ", inTracker2: " << inTracker2 << "\n";

    std::wcerr << "Copying from host -> device " << devno1 << " -> device " << devno2 << " -> host:\n";
    
    SHOW_TIME(acc_view1.copy_async(host_data1.data(), device_data1.accelerator_pointer(), size * sizeof(double)));
    // SHOW_TIME(acc_view1.copy_async(device_data1.accelerator_pointer(), device_data2.accelerator_pointer(),
    //                                size * sizeof(double)));
    SHOW_TIME(acc_view1.wait());
    try {
      SHOW_TIME(acc_view1.copy_ext(device_data1.accelerator_pointer(), device_data2.accelerator_pointer(),
				   size * sizeof(double), hcMemcpyDeviceToDevice,
				   devPtrInfo1, devPtrInfo2, &acc1, false));
      SHOW_TIME(acc_view1.wait());
    }
    catch(std::exception& e){
      std::cerr << "Exception caught: " << e.what() << '\n';
    }
    SHOW_TIME(acc_view2.copy_async(device_data2.accelerator_pointer(), host_data2.data(), size * sizeof(double)).wait());
    
    auto average2 = std::accumulate(host_data2.begin(), host_data2.end(), 0.0) / size;
    std::wcerr << "average2: " << average2 << '\n';
    
    am_free(device_ptr1);
    am_free(device_ptr2);
  }
  std::wcerr << "total time: " << tm << " seconds\n";
}
