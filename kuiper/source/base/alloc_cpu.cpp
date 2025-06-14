#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define KUIPER_HAVE_POSIX_MEMALIGN
#endif//件编译判断系统是否支持 posix_memalign，这是一个对齐内存分配函数。如果支持就定义 KUIPER_HAVE_POSIX_MEMALIGN 宏。

namespace base {
    CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
    }

    void *CPUDeviceAllocator::allocate(size_t byte_size) const {
        if (!byte_size) {
            return nullptr;
        }
#ifdef KUIPER_HAVE_POSIX_MEMALIGN
        void *data = nullptr;
        const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);//如果内存大于等于 1KB，使用 32 字节对齐。 否则使用16字节对齐
        //使用 posix_memalign 分配内存 第一个地址是结果地址的地址 ，第二个是对齐 大小，第三个是实际分配字节数
        int status = posix_memalign((void **) &data,
                                    ((alignment >= sizeof(void *)) ? alignment : sizeof(void *)),
                                    byte_size);
        if (status != 0) {
            return nullptr;
        }
        return data;
#else
        void* data = malloc(byte_size);
        return data;
#endif
    }

    void CPUDeviceAllocator::release(void *ptr) const {
        if (ptr) {
            free(ptr);//调用标准C的free释放内存
        }
    }

    std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
}  // namespace base