#ifndef KUIPER_INCLUDE_BASE_ALLOC_H_
#define KUIPER_INCLUDE_BASE_ALLOC_H_

#include <map>
#include <memory>
#include "base.h"

namespace base {
    enum class MemcpyKind {//枚举类型 表示拷贝类型
        kMemcpyCPU2CPU = 0,
        kMemcpyCPU2CUDA = 1,
        kMemcpyCUDA2CPU = 2,
        kMemcpyCUDA2CUDA = 3,
    };

    class DeviceAllocator {
    public:
        explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

        virtual DeviceType device_type() const { return device_type_; }

        virtual void release(void *ptr) const = 0;

        virtual void *allocate(size_t byte_size) const = 0;

        virtual void memcpy(const void *src_ptr, void *dest_ptr, size_t byte_size,
                            MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void *stream = nullptr,
                            bool need_sync = false) const;

        virtual void memset_zero(void *ptr, size_t byte_size, void *stream, bool need_sync = false);

    private:
        DeviceType device_type_ = DeviceType::kDeviceUnknown;
    };

    class CPUDeviceAllocator : public DeviceAllocator {
    public:
        explicit CPUDeviceAllocator();

        void *allocate(size_t byte_size) const override;

        void release(void *ptr) const override;
    };

    struct CudaMemoryBuffer {
        void *data;// 指向内存的起始地址
        size_t byte_size;
        bool busy;

        CudaMemoryBuffer() = default;

        CudaMemoryBuffer(void *data, size_t byte_size, bool busy)
                : data(data), byte_size(byte_size), busy(busy) {}
    };

    class CUDADeviceAllocator : public DeviceAllocator {
    public:
        explicit CUDADeviceAllocator();

        void *allocate(size_t byte_size) const override;//const表示这个成员函数不会修改当前对象的任何成员变量。 若前边有const 则修饰返回值

        void release(void *ptr) const override;

    private:
        mutable std::map<int, size_t> no_busy_cnt_;
        mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;//mutable 是为了让这些成员变量在 const 函数中也能被修改 . const在函数后 代表承诺不修改这个类的成员变量
        mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
    };

    class CPUDeviceAllocatorFactory {
    public:
        static std::shared_ptr<CPUDeviceAllocator> get_instance() {//经典单例模式实现
            if (instance == nullptr) {
                instance = std::make_shared<CPUDeviceAllocator>();
            }
            return instance;
        }

    private:
        static std::shared_ptr<CPUDeviceAllocator> instance;//private只能在类的内部访问
    };

    class CUDADeviceAllocatorFactory {
    public:
        static std::shared_ptr<CUDADeviceAllocator> get_instance() {
            if (instance == nullptr) {
                instance = std::make_shared<CUDADeviceAllocator>();
            }
            return instance;
        }

    private:
        static std::shared_ptr<CUDADeviceAllocator> instance;
    };
}  // namespace base
#endif  // KUIPER_INCLUDE_BASE_ALLOC_H_