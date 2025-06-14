#include "base/buffer.h"
#include <glog/logging.h>

namespace base {
    Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void *ptr,
                   bool use_external)
            : byte_size_(byte_size),
              allocator_(allocator),
              ptr_(ptr),
              use_external_(use_external) {
        if (!ptr_ && allocator_) {
            device_type_ = allocator_->device_type();
            use_external_ = false;
            ptr_ = allocator_->allocate(byte_size);
        }
    }

    Buffer::~Buffer() {
        if (!use_external_) {//如果我们这里将use_external置为false，表示当前Buffer拥有该内存，表示这块资源需要Buffer进行管理，那么在Buffer对象释放的时候会调用对应allocator的释放方法，自动释放这块内存
            if (ptr_ && allocator_) {
                allocator_->release(ptr_);
                ptr_ = nullptr;
            }
        }
    }

    void *Buffer::ptr() {
        return ptr_;
    }

    const void *Buffer::ptr() const {
        return ptr_;
    }

    size_t Buffer::byte_size() const {
        return byte_size_;
    }

    bool Buffer::allocate() {
        if (allocator_ && byte_size_ != 0) {
            use_external_ = false; //说明buffer会对内存负责
            ptr_ = allocator_->allocate(byte_size_);
            if (!ptr_) {
                return false;
            } else {
                return true;
            }
        } else {
            return false;
        }
    }

    std::shared_ptr<DeviceAllocator> Buffer::allocator() const {
        return allocator_;
    }

    void Buffer::copy_from(const Buffer &buffer) const {//引用版本
        CHECK(allocator_ != nullptr);
        CHECK(buffer.ptr_ != nullptr);
        //如果当前对象（this）的 byte_size_ 小于 buffer.byte_size_，那么就使用当前对象的大小；否则使用对方 buffer 的大小
        size_t byte_size = byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;
        const DeviceType &buffer_device = buffer.device_type(); //获取 源 buffer 的设备类型（是 CPU 还是 CUDA？） 获取的引用类型 避免拷贝
        const DeviceType &current_device = this->device_type();
        CHECK(buffer_device != DeviceType::kDeviceUnknown &&
              current_device != DeviceType::kDeviceUnknown);

        if (buffer_device == DeviceType::kDeviceCPU &&
            current_device == DeviceType::kDeviceCPU) {//普通内存拷贝：从一个 CPU buffer 拷贝到另一个。相当于使用std::memcpy()
            return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size);
        } else if (buffer_device == DeviceType::kDeviceCUDA &&
                   current_device == DeviceType::kDeviceCPU) {
            return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size,
                                      MemcpyKind::kMemcpyCUDA2CPU);//GPU到CPU
        } else if (buffer_device == DeviceType::kDeviceCPU &&
                   current_device == DeviceType::kDeviceCUDA) {
            return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size,
                                      MemcpyKind::kMemcpyCPU2CUDA);//CPU → CUDA
        } else {
            return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size,
                                      MemcpyKind::kMemcpyCUDA2CUDA);
        }
    }

    void Buffer::copy_from(const Buffer *buffer) const {//指针版本
        CHECK(allocator_ != nullptr);
        CHECK(buffer != nullptr || buffer->ptr_ != nullptr);

        size_t dest_size = byte_size_;
        size_t src_size = buffer->byte_size_;
        size_t byte_size = src_size < dest_size ? src_size : dest_size;

        const DeviceType &buffer_device = buffer->device_type();
        const DeviceType &current_device = this->device_type();
        CHECK(buffer_device != DeviceType::kDeviceUnknown &&
              current_device != DeviceType::kDeviceUnknown);

        if (buffer_device == DeviceType::kDeviceCPU &&
            current_device == DeviceType::kDeviceCPU) {
            return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size);
        } else if (buffer_device == DeviceType::kDeviceCUDA &&
                   current_device == DeviceType::kDeviceCPU) {
            return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                                      MemcpyKind::kMemcpyCUDA2CPU);
        } else if (buffer_device == DeviceType::kDeviceCPU &&
                   current_device == DeviceType::kDeviceCUDA) {
            return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                                      MemcpyKind::kMemcpyCPU2CUDA);
        } else {
            return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                                      MemcpyKind::kMemcpyCUDA2CUDA);
        }
    }

    DeviceType Buffer::device_type() const {
        return device_type_;
    }

    void Buffer::set_device_type(DeviceType device_type) {
        device_type_ = device_type;
    }

    std::shared_ptr<Buffer> Buffer::get_shared_from_this() {
        return shared_from_this();
    }

    bool Buffer::is_external() const {
        return this->use_external_;
    }

}  // namespace base