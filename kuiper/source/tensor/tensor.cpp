#include "tensor/tensor.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <numeric>

namespace tensor {
    template<typename T, typename Tp>//函数功能是取出所有的值 相乘 得到积
    static size_t reduce_dimension(T begin, T end, Tp init) {
        if (begin >= end) {//如果begin==end 说明没有任何维度 返回0
            return 0;
        }
        //std::accumulate：标准库算法，用于累加（或其他操作）一段迭代器范围内的元素；  init初始值 通常为1；std::multiplies<>()：乘法运算函数对象；
        size_t size = std::accumulate(begin, end, init, std::multiplies<>());
        return size;
    }

    static size_t data_type_size(base::DataType data_type) {//返回该类型的字节数
        switch (data_type) {
            case base::DataType::kDataTypeFp32: {
                return 4;
            }
            case base::DataType::kDataTypeInt8: {
                return 1;
            }
            case base::DataType::kDataTypeInt32: {
                return 4;
            }
            default: {
                LOG(FATAL) << "Unknown data type size for " << int(data_type);
                return 0;
            }
        }
    }

    Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc, void *ptr)
            : data_type_(data_type) {
        dims_.push_back(dim0);
        size_ = dim0;
        if (need_alloc && alloc) {
            //这里形参有俩 为啥这里调用只有一个？因为.h文件里有默认值
            //为啥没写this? 会先从当前类中找，如果有就等价于this->
            allocate(alloc);
//            this->allocate(alloc);
        } else {
            if (ptr != nullptr) {
                CHECK(need_alloc == false)
                                << "The need_alloc is is true when ptr parameter is not a null pointer.";
                init_buffer(alloc, data_type_, need_alloc, ptr);
            }
        }
    }

    Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc, void *ptr)
            : data_type_(data_type) {
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        size_ = dim0 * dim1;
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            init_buffer(alloc, data_type_, need_alloc, ptr);
        }
    }

    Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc, void *ptr)
            : data_type_(data_type) {
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        dims_.push_back(dim2);
        size_ = dim0 * dim1 * dim2;
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            init_buffer(alloc, data_type_, need_alloc, ptr);
        }
    }

    Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                   bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc, void *ptr)
            : data_type_(data_type) {
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        dims_.push_back(dim2);
        dims_.push_back(dim3);
        size_ = dim0 * dim1 * dim2 * dim3;
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            init_buffer(alloc, data_type_, need_alloc, ptr);
        }
    }

    Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc, void *ptr)
            : dims_(std::move(dims)), data_type_(data_type) {
        size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            init_buffer(alloc, data_type_, need_alloc, ptr);
        }
    }

    void Tensor::to_cuda(cudaStream_t stream) {
        CHECK_NE(buffer_, nullptr);
        const base::DeviceType device_type = this->device_type();
        if (device_type == base::DeviceType::kDeviceUnknown) {
            LOG(ERROR) << "The device type of the tensor is unknown.";
        } else if (device_type == base::DeviceType::kDeviceCPU) {
            size_t byte_size = this->byte_size();
            auto cu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
            auto cu_buffer = std::make_shared<base::Buffer>(byte_size, cu_alloc);
            cu_alloc->memcpy(buffer_->ptr(), cu_buffer->ptr(), byte_size, base::MemcpyKind::kMemcpyCPU2CUDA,
                             stream);
            this->buffer_ = cu_buffer;
        } else {
            LOG(INFO) << "The device type of the tensor is already cuda.";
        }
    }

    void Tensor::to_cpu() {
        CHECK_NE(buffer_, nullptr);
        const base::DeviceType device_type = this->device_type();

        if (device_type == base::DeviceType::kDeviceUnknown) {
            LOG(ERROR) << "The device type of the tensor is unknown.";
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            size_t byte_size = this->byte_size();
            auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
            auto cpu_buffer = std::make_shared<base::Buffer>(byte_size, cpu_alloc);
            cpu_alloc->memcpy(buffer_->ptr(), cpu_buffer->ptr(), byte_size,
                              base::MemcpyKind::kMemcpyCUDA2CPU);
            this->buffer_ = cpu_buffer;
        } else {
            LOG(INFO) << "The device type of the tensor is already cpu.";
        }
    }

    size_t Tensor::size() const { return this->size_; }

    int32_t Tensor::get_dim(int32_t idx) const {
        CHECK_GE(idx, 0);// 断言 idx >= 0
        CHECK_LT(idx, this->dims_.size());// 断言 idx < dims_ 的大小
        return this->dims_.at(idx);
    }

    base::DeviceType Tensor::device_type() const {
        if (!buffer_) {
            return base::DeviceType::kDeviceUnknown;
        }
        return buffer_->device_type();
    }

    bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
        if (!buffer) {
            LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
            return false;
        }
        if (buffer_) {
            if (buffer_->device_type() != buffer->device_type()) {
                LOG(ERROR) << "The device type of the new buffer is different from the original one.";
            }
        }

        size_t byte_size = this->byte_size();
        if (byte_size > buffer->byte_size()) {
            LOG(ERROR) << "The size of buffer is too small for the tensor!";
            return false;
        }
        buffer_ = buffer;
        return true;
    }
    // bool need_realloc = false ,need_alloc表示是否需要用内存分配器alloc来分配内存/显存
    //控制了是否用指定的分配器为当前Tensor分配内存 或 显存
    bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
        if (!allocator) {
            LOG(ERROR) << "The allocator parameter in the allocate function is null "
                          "pointer!";
            return false;
        }

        size_t byte_size = this->byte_size();//拿当前tensor应该占的内存大小
        if (!byte_size) {
            LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
            return false;
        }

        if (buffer_ && byte_size <= buffer_->byte_size()) {
            //如果当前已经有分配好的buffer_ 并且 已经足够大了 ，就复用
            if (!need_realloc) {
                return true;//复用原来的内存 不需要重新分配
            }
        }

        buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);//创建一个新的buffer对象
        if (!buffer_->ptr()) {
            LOG(ERROR) << "The memory allocated is a null pointer!";
            return false;
        }
        return true;
    }

    const std::vector<int32_t> &Tensor::dims() const { return this->dims_; }

    void Tensor::set_device_type(base::DeviceType device_type) const {
        if (buffer_) {
            buffer_->set_device_type(device_type);
        }
    }

    void Tensor::reset(base::DataType data_type, const std::vector<int32_t> &dims) {
        this->data_type_ = data_type;
        this->dims_ = dims;
        this->size_ = reduce_dimension(dims.begin(), dims.end(), 1);
        this->buffer_ = nullptr;
    }

    int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }

    base::DataType Tensor::data_type() const { return data_type_; }

    void Tensor::reshape(const std::vector<int32_t> &dims) {
        size_t size = reduce_dimension(dims.begin(), dims.end(), 1);
        if (!buffer_) {//如果当前没有任何数据（buffer为空） 就只修改形状信息即可，不需要再考虑内存问题
            this->dims_ = dims;
            this->size_ = size;
            return;
        }

        if (size > size_) {
            auto new_buffer = std::make_shared<base::Buffer>(size * base::DataTypeSize(this->data_type_),
                                                             buffer_->allocator());
            CHECK(new_buffer->allocate());
            new_buffer->copy_from(buffer_.get());//buffer_是一个std::shared_ptr<base::Buffer>，这是一个引用计数的智能指针。 没有计数会自动释放
            this->buffer_ = new_buffer;
        }
        this->dims_ = dims;
        this->size_ = size;
    }

    std::shared_ptr<base::Buffer> Tensor::get_buffer() const { return buffer_; }

    Tensor Tensor::clone() const {//创建一个内容完全相同的副本 （深拷贝）
        Tensor new_tensor = *this;//所有成员变量都会被浅拷贝（例如 dims_、data_type_、buffer_ 等）。 这时候 new_tensor.buffer_ 和当前的 this->buffer_ 指向的是同一块内存（共享）。
        size_t byte_size = this->byte_size();

        auto allocator = buffer_->allocator();
        new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);//为新 tensor 分配一个新的内存 buffer，不再共享旧内存，实现真正的“深拷贝”。
        new_tensor.buffer_->copy_from(buffer_.get());//将当前tensor的内存内容拷贝到新buffer中 确保数据独立
        return new_tensor;
    }

    size_t Tensor::byte_size() const { return this->size() * DataTypeSize(data_type_); }

    std::vector<size_t> Tensor::strides() const {//计算各个维度的步长
        // 举例：3D 张量 [2, 3, 4]（对应 dim0, dim1, dim2）
        //这个张量有：
        //dim0 = 2
        //dim1 = 3
        //dim2 = 4
        //你按顺序访问顺序是：
        //[0][0][0], [0][0][1], ..., [0][1][0], ..., [1][2][3]（一维连续内存）
        //stride 的计算：
        //stride[0] = 3 × 4 = 12（跳过一个 dim0，要跳过12个元素）
        //stride[1] = 4 （跳过一个 dim1，要跳过4个元素）
        //stride[2] = 1 （dim2 连续排布）
        std::vector<size_t> strides;
        if (!dims_.empty()) {
            for (int32_t i = 0; i < dims_.size() - 1; ++i) {//最后一维不用算 肯定是1
                size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1);
                strides.push_back(stride);
            }
            strides.push_back(1);
        }
        return strides;
    }

    bool Tensor::is_empty() const {
        return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
    }

    void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                             bool need_alloc, void *ptr) {
        if (!alloc && !need_alloc) { //如果没传alloc 并且也不需要自己分配内存，那么说明已经有了外部内存ptr
            std::shared_ptr<base::Buffer> buffer =
                    std::make_shared<base::Buffer>(data_type_size(data_type) * size_, nullptr, ptr, true);
            this->buffer_ = buffer;
        } else {
            allocate(alloc, true);
        }
    }
}  // namespace tensor