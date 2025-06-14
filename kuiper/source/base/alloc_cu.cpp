#include <cuda_runtime_api.h>
#include "base/alloc.h"

namespace base {

    CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

    void *CUDADeviceAllocator::allocate(size_t byte_size) const {
        //一个cuda内存分配器的实现函数 根据申请内存的大小 在已有的内存池种复用空闲内存 如果没有合适的，就调用cudaMalloc()重新申请一块
        int id = -1;
        cudaError_t state = cudaGetDevice(&id);//获取当前使用的 GPU 设备 ID。
        CHECK(state == cudaSuccess);
        if (byte_size > 1024 * 1024) { //大内存处理逻辑 进入大内存池逻辑 1MB以上被认为是大内存块

            //查找是否有空闲大buffer可复用   mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_
            auto &big_buffers = big_buffers_map_[id];
            int sel_id = -1;
            for (int i = 0; i < big_buffers.size(); i++) {//遍历当前设备的big_buffers_map_
                if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
                    big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024) {
                    // 找到空闲的（！busy）  大小够用（>=byte_size）  且浪费不要太多（冗余小于1MB） 最后选择能满足要求的块
                    if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
                        sel_id = i;
                    }
                }
            }
            if (sel_id != -1) { //若找到可复用的块
                big_buffers[sel_id].busy = true;
                return big_buffers[sel_id].data;
            }

            void *ptr = nullptr;
            //为什么还要取址？
            //cudaError_t cudaMalloc(void** devPtr, size_t size);
            state = cudaMalloc(&ptr, byte_size);
            if (cudaSuccess != state) {
                //snprintf 是 C/C++ 中的一个格式化字符串函数，作用是：把格式化好的字符串写入一个字符数组（而不是直接打印出来）
                char buf[256];
                snprintf(buf, 256,
                         "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
                         "left on  device.",//使用 snprintf 构造格式化字符串（比如“申请了 64 MB 内存失败”）
                         byte_size >> 20);//将字节数除以 2^20（即 1024 * 1024），换算成 MB 单位；
                LOG(ERROR) << buf;
                return nullptr;
            }
            big_buffers.emplace_back(ptr, byte_size, true);//如果显存申请成功，创建一个新的 CudaMemoryBuffer 对象并加入内存池 big_buffers：
            return ptr;
        }

        auto &cuda_buffers = cuda_buffers_map_[id];
        for (int i = 0; i < cuda_buffers.size(); i++) {
            if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
                cuda_buffers[i].busy = true;
                no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
                return cuda_buffers[i].data;
            }
        }
        void *ptr = nullptr;
        state = cudaMalloc(&ptr, byte_size);
        if (cudaSuccess != state) {
            char buf[256];
            snprintf(buf, 256,
                     "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
                     "left on  device.",
                     byte_size >> 20);
            LOG(ERROR) << buf;
            return nullptr;
        }
        cuda_buffers.emplace_back(ptr, byte_size, true);
        return ptr;
    }

    void CUDADeviceAllocator::release(void *ptr) const {
        if (!ptr) {
            return;
        }
        if (cuda_buffers_map_.empty()) {
            return;
        }
        cudaError_t state = cudaSuccess;
        for (auto &it: cuda_buffers_map_) {
            if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024) {
                auto &cuda_buffers = it.second;
                std::vector<CudaMemoryBuffer> temp;
                for (int i = 0; i < cuda_buffers.size(); i++) {
                    if (!cuda_buffers[i].busy) {
                        state = cudaSetDevice(it.first);
                        state = cudaFree(cuda_buffers[i].data);
                        CHECK(state == cudaSuccess)
                                        << "Error: CUDA error when release memory on device " << it.first;
                    } else {
                        temp.push_back(cuda_buffers[i]);
                    }
                }
                cuda_buffers.clear();
                it.second = temp;
                no_busy_cnt_[it.first] = 0;
            }
        }

        for (auto &it: cuda_buffers_map_) {
            auto &cuda_buffers = it.second;
            for (int i = 0; i < cuda_buffers.size(); i++) {
                if (cuda_buffers[i].data == ptr) {
                    no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
                    cuda_buffers[i].busy = false;
                    return;
                }
            }
            auto &big_buffers = big_buffers_map_[it.first];
            for (int i = 0; i < big_buffers.size(); i++) {
                if (big_buffers[i].data == ptr) {
                    big_buffers[i].busy = false;
                    return;
                }
            }
        }
        state = cudaFree(ptr);
        CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
    }

    std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

}  // namespace base