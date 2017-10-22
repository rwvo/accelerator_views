#include <hc.hpp>
#include <hc_am.hpp>
#include <pinned_vector.hpp>
#include <iostream>

using namespace hc;

constexpr size_t size = 64;
#define TEST_COPY(copy_operation)					\
  std::cerr << #copy_operation;						\
  try {									\
    auto fut = copy_operation;						\
    acc_view.wait();							\
    std::cerr << " PASSED\n";						\
  }									\
  catch(std::exception& e){						\
    std::cerr << " FAILED\n";						\
    std::cerr << e.what() << '\n';					\
  }



int main(){
  hc::accelerator acc;
  auto acc_view = acc.get_default_view();


  pinned_vector<int> host_data(size); // host pinned memory, alloc-ed with am_alloc, zero-initialized
  auto device_ptr = hc::am_alloc(size * sizeof(int), acc, 0);
  auto device_data = hc::array<int, 1>(extent<1>(size), acc_view, device_ptr);
  std::cout << "device_ptr: " << (void*)device_ptr << '\n';

  /*
  hc::AmPointerInfo hostPtrInfo(NULL, NULL, 0, acc, 0, 0);
  hc::AmPointerInfo devtrInfo(NULL, NULL, 0, acc, 0, 0);
  
  bool hostmemInTracker (hc::am_memtracker_getinfo(&hostPtr Info, host_data.data()) == AM_SUCCESS); 
  bool devmemInTracker (hc::am_memtracker_getinfo(&hostPtrInfo, device_data.data()) == AM_SUCCESS);

  std::cout << "hostmemInTracker: " << hostmemInTracker << ", devmemInTracker: " << devmemInTracker << '\n';
  */
  
  // hc::copy_async, host to device
  try {
    auto fut = hc::copy_async(host_data.data(), device_data);
    acc_view.wait();
    fut.get();
    std::cerr << "hc::copy_async(host_data.data(), device_data) PASSED\n";
  }
  catch(std::exception& e){
    std::cerr << "hc::copy_async(host_data.data(), device_data) FAILED\n";
    std::cerr << e.what() << '\n';
  }

  // hc::copy_async, device to host
  try {
    auto fut = hc::copy_async(device_data, host_data.data());
    acc_view.wait();
    fut.get();
    std::cerr << "hc::copy_async(device_data, host_data.data()) PASSED\n";
  }
  catch(std::exception& e){
    std::cerr << "hc::copy_async(device_data, host_data.data()) FAILED\n";
    std::cerr << e.what() << '\n';
  }

  TEST_COPY(acc_view.copy_async(host_data.data(), device_ptr, size * sizeof(int)));
  TEST_COPY(acc_view.copy_async(host_data.data(), device_ptr, size * sizeof(int)));
  TEST_COPY(acc_view.copy_async(host_data.data(), device_ptr, size * sizeof(int)));

  TEST_COPY(acc_view.copy_async(host_data.data(), device_data.accelerator_pointer(), size * sizeof(int)));
  TEST_COPY(acc_view.copy_async(host_data.data(), device_data.accelerator_pointer(), size * sizeof(int)));
  TEST_COPY(acc_view.copy_async(host_data.data(), device_data.accelerator_pointer(), size * sizeof(int)));

  TEST_COPY(acc_view.copy_async(device_data.accelerator_pointer(), host_data.data(), size * sizeof(int)));
  TEST_COPY(acc_view.copy_async(device_data.accelerator_pointer(), host_data.data(), size * sizeof(int)));
  TEST_COPY(acc_view.copy_async(device_data.accelerator_pointer(), host_data.data(), size * sizeof(int)));

  /*
  

  am_free(device_ptr);
  */
  std::cerr << "leaving main()\n";
}
