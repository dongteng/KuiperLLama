#include "base/alloc.h"
#include <cuda_runtime_api.h>


namespace base {
    void DeviceAllocator::memcpy(const void *src_ptr, void *dest_ptr, size_t byte_size,
                                 MemcpyKind memcpy_kind, void *stream, bool need_sync) const {
//        用于在不同内存之间拷贝数据
// need_sync：是否需要在拷贝后调用 cudaDeviceSynchronize() 来同步。
//为什么 void *stream 要用 void * 类型？  使用 void* stream 是为了在函数接口上保持通用性与跨平台、跨后端的灵活性。
        CHECK_NE(src_ptr, nullptr);
        CHECK_NE(dest_ptr, nullptr);//目的地址指针
        if (!byte_size) {
            return;
        }

        cudaStream_t stream_ = nullptr;//CUDA流 异步执行时使用
        if (stream) {
            stream_ = static_cast<CUstream_st *>(stream);//如果传入了 stream，就强制转换成 cudaStream_t 类型；否则默认使用空流（即同步拷贝）
        }
        if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
            std::memcpy(dest_ptr, src_ptr, byte_size);//用标准库函数完成主机内存之间的数据拷贝
        } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) { //cpu -> cuda
            if (!stream_) {
                cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);//没有传入流 则使用同步的cudaMemcy
            } else {
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
            }
        } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) { //cuda->cpu
            if (!stream_) {
                cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
            } else {
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
            }
        } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
            if (!stream_) {
                cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
            } else {
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
            }
        } else {
            LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
        }
        if (need_sync) {
            cudaDeviceSynchronize();//如果用户指定需要同步，那么调用 cudaDeviceSynchronize() 来等待所有设备上的操作完成。
        }
    }

    void DeviceAllocator::memset_zero(void *ptr, size_t byte_size, void *stream,
                                      bool need_sync) {
        //ptr：指向要清零的内存地址（可以是 CPU 也可以是 GPU）
        CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
        if (device_type_ == base::DeviceType::kDeviceCPU) {
            std::memset(ptr, 0, byte_size);//如果是CPU设备 调用标准库函数直接清零
        } else {
            if (stream) {
                cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
                cudaMemsetAsync(ptr, 0, byte_size, stream_);
            } else {
                cudaMemset(ptr, 0, byte_size);
            }
            if (need_sync) {
                cudaDeviceSynchronize();
            }
        }
    }

}  // namespace base