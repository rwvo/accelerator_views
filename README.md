## hc::accelerator_view, overlapping transfers and computation, device-to-device DMA transfers.
### Initial experiments.

Messy code lives here. Code will be sanitized and commented soon.

### Some quick observations

* The global `hc::copy_async` and the member function `accelerator_view::copy_async` don't play together nicely. The
  former supposedly uses the default accelerator_view, but even if I get the default accelerator_view and use the member
  function on that view, mixing the two frequently results in crashes. When working with multiple queues, it seems best
  to only use the `copy_async` member function.

* Asynchronous operations submitted to an accelerator_view (i.e., copy_async and parallel_for_each) are executed
  sequentially: an operation only starts when the previous one has finished. Only the *submission* is asynchronous
  (returns immediately, before the submitted operation has finished). This is a good thing; it would be very hard to get
  overlapping work on two or more accelerator_views going otherwise. In some of the example code, I do explicit
  `wait()`s on the operations, but that is just to time the actual execution of the operations. In practice, if you'd do
  a copy from host to device, followed by a bunch of parallel_for_each ops, followed by a copy from device to host,
  you'd only have to `wait()` on the last copy before processing the results on the host. You can either `wait()` on the
  `completion_future` returned from the `copy_async`, or on the `accelerator_view` itself.

* Unlike the global `hc::copy_async``, the member function `accelerator_view::copy_async` requires that both the source
  and the target pointer are allocated with `am_alloc`. Big shout-out to Scott Moe for discovering that! That's somewhat
  unfortunate if you want to use `hc::array` to represent data on the device. Fortunately, there is an ``hc::array``
  constructor that takes a raw pointer to already `am_alloc`-ed memory, as is shown in the code examples. Unfortunately,
  the array doesn't take ownership of the memory; you have to delete it yourself, the array doesn't do it when its
  lifetime end. That can be a rich source of bugs in larger codes with complicated code paths, but it should be possible
  to wrap the am_alloced pointer in a `std::unique_ptr` or a `std::shared_pointer`, and provide a custom deleter,
  similar to how hc::pinned_vector handles am_alloced host-pinned memory. I'll provide some examples in a next iteration.


### Overlapping transfers/computations on two accelerator_views

See code under [04_multi_acc_view_copy]. Creates two accelerator_views, copies data (all zeros) to both of them, does a
whole lot of computation (on both queueus) resulting in all 1.0 values in both arrays on the device, copies the results
back, and checks that the average value is 1.0 indeed for both vectors on the host.

Commenting out the `acc_view1.wait()` between the work on the two queues results in a total time for sequential execution.

This code does not test for overlap of transfer on the one hand, and computation on the other hand. Instead, it tests
overlap of two independent transfer-compute-transfer sequences.

### DMA transfers between two GPU devices.

See code under [05_multi_device_acc_view_copy]. The magic for avoiding segfault/core dumps is a call to
`hc::am_map_to_peers`. First arg: device pointer that is to be mapped to other accelerators. Third arg: pointer/array of
`hc::accelerators` to which the pointer needs to be mapped. Second arg: number of accelerators in third arg.

Fill host array with multiple copies of Pi, copy to device 1, copy from device 1 to device 2, copy from device 2 to
host, check that the average value in the resulting host array is still Pi.

In the current version, I do a `wait()` on all copies, to see the individual times. This appears to also properly
synchronize between the devices: device2 should not start the copy back to the host before device1 finishes the copy to
device2, and device2 can actually see the results. A better way of synchronization using markers is shown in a unit test
of the ROCm source code: `hcc/tests/Unit/AcceleratorViewCopy/copy_coherency.cpp`, search for `am_map_to_peers`.

Tested it on a P47 system (EPIC with 4 MI25 GPUs), and the device-to-device transfer doesn't give stellar performance yet:

```
acc_view1.copy_async(host_data1.data(), device_data1.accelerator_pointer(), size * sizeof(double)).wait(): 0.081704 seconds, 12.2393GiB/s
acc_view1.copy_async(device_data1.accelerator_pointer(), device_data2.accelerator_pointer(), size * sizeof(double)).wait(): 0.234224 seconds, 4.26942GiB/s
acc_view2.copy_async(device_data2.accelerator_pointer(), host_data2.data(), size * sizeof(double)).wait(): 0.10079 seconds, 9.92162GiB/s
```

From these number, it looks like a copy from device1 to host, and from host to device2, should be faster than the direct
copy from device1 to device2, which is dissapointing. `copy_async_ext` did not result in better performance. But at
least we went from a copy operation that crashes fast to one that succeeds somewhat slowly.

