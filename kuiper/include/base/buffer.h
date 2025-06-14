#ifndef KUIPER_INCLUDE_BASE_BUFFER_H_
#define KUIPER_INCLUDE_BASE_BUFFER_H_

#include <memory>
#include "base/alloc.h"

namespace base {
    class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
    private:
        size_t byte_size_ = 0;//这块内存的大小 以字节数为单位
        void *ptr_ = nullptr; //这块内存地址 有2种来源 1是外部直接赋值得到 buffer不需要对它进行管理 和它的关系是借用，不负责它的生命周期管理，这种情况下对应下方use_external的值置为true。
        bool use_external_ = false;//3. 另外一种是需要Buffer对这块内存进行管理的，所以use_external值为false，表示需要对它的生命周期进行管理，也就是没人使用该Buffer的时候会自动将ptr_指向的地址用对应类型的Allocator完成释放。
        DeviceType device_type_ = DeviceType::kDeviceUnknown;//表示Buffer中内存资源所属的设备类型
        std::shared_ptr<DeviceAllocator> allocator_;//Buffer对应设备类型的内存分配器，我们已经在上一节中说过，负责资源的释放、申请以及拷贝等，既可以是cpu allocator 也可以是cuda allocator.

    public:
        explicit Buffer() = default;

        explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                        void *ptr = nullptr, bool use_external = false);

        virtual ~Buffer();

        bool allocate();

        void copy_from(const Buffer &buffer) const;

        void copy_from(const Buffer *buffer) const;

        void *ptr();

        const void *ptr() const;

        size_t byte_size() const;

        std::shared_ptr<DeviceAllocator> allocator() const;

        DeviceType device_type() const;

        void set_device_type(DeviceType device_type);

        std::shared_ptr<Buffer> get_shared_from_this();

        bool is_external() const;
    };
}  // namespace base

#endif