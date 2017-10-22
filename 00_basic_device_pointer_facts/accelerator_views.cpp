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
  {
    auto device_data = hc::array<int, 1>(extent<1>(size), acc_view, device_ptr);
    auto device_data2 = hc::array<int, 1>(extent<1>(size));
    std::cout << "device_ptr: " << (void*)device_ptr << '\n';
    std::cout << "device_data.data(): " << device_data.data() << '\n';
    std::cout << "device_data.accelerator_pointer(): " << device_data.accelerator_pointer() << '\n';
    std::cout << "device_data2.accelerator_pointer(): " << device_data2.accelerator_pointer() << '\n';
    
    
    hc::AmPointerInfo devPtrInfo(NULL, NULL, NULL, 0, acc, 0, 0);
    bool inTracker = hc::am_memtracker_getinfo(&devPtrInfo, device_ptr) == AM_SUCCESS;
    std::cout << "device_ptr in tracker: " << inTracker << '\n';
    inTracker = hc::am_memtracker_getinfo(&devPtrInfo, device_data.accelerator_pointer()) == AM_SUCCESS;
    std::cout << "device_data.accelerator_pointer() in tracker : " << inTracker << '\n';
    inTracker = hc::am_memtracker_getinfo(&devPtrInfo, device_data2.accelerator_pointer()) == AM_SUCCESS;
    std::cout << "device_data2.accelerator_pointer() in tracker : " << inTracker << '\n';
    
    TEST_COPY(acc_view.copy_async(host_data.data(), device_ptr, size * sizeof(int)));
    TEST_COPY(acc_view.copy_async(host_data.data(), device_ptr, size * sizeof(int)));
    TEST_COPY(acc_view.copy_async(host_data.data(), device_ptr, size * sizeof(int)));
    
    TEST_COPY(acc_view.copy_async(host_data.data(), device_data.accelerator_pointer(), size * sizeof(int)));
    TEST_COPY(acc_view.copy_async(host_data.data(), device_data.accelerator_pointer(), size * sizeof(int)));
    TEST_COPY(acc_view.copy_async(host_data.data(), device_data.accelerator_pointer(), size * sizeof(int)));
    
    TEST_COPY(acc_view.copy_async(device_data.accelerator_pointer(), host_data.data(), size * sizeof(int)));
    TEST_COPY(acc_view.copy_async(device_data.accelerator_pointer(), host_data.data(), size * sizeof(int)));
    TEST_COPY(acc_view.copy_async(device_data.accelerator_pointer(), host_data.data(), size * sizeof(int)));
    
    inTracker = (hc::am_memtracker_getinfo(&devPtrInfo, device_ptr) == AM_SUCCESS);
    std::cout << "device_ptr in tracker: " << inTracker << '\n';
    std::cerr << "leaving inner scope()\n";
  }
  hc::AmPointerInfo devPtrInfo(NULL, NULL, NULL, 0, acc, 0, 0);
  bool inTracker = hc::am_memtracker_getinfo(&devPtrInfo, device_ptr) == AM_SUCCESS;
  std::cout << "device_ptr in tracker: " << inTracker << '\n';
  am_free(device_ptr);
  inTracker = (hc::am_memtracker_getinfo(&devPtrInfo, device_ptr) == AM_SUCCESS);
  std::cout << "device_ptr in tracker: " << inTracker << '\n';
}
