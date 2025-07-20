#ifndef KUIPER_INCLUDE_TENSOR_TENSOR_H_
#define KUIPER_INCLUDE_TENSOR_TENSOR_H_

#include <driver_types.h>
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <vector>
#include "base/base.h"
#include "base/buffer.h"

namespace tensor {

    class Tensor {
    public:
        explicit Tensor() = default;

        explicit Tensor(base::DataType data_type, int32_t dim0, bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void *ptr = nullptr);

        explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void *ptr = nullptr);

        explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                        bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                        void *ptr = nullptr);

        explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                        bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                        void *ptr = nullptr);

        explicit Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void *ptr = nullptr);

        void to_cpu();

        void to_cuda(cudaStream_t stream = nullptr);

        bool is_empty() const;

        void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                         bool need_alloc, void *ptr);

        template<typename T>
        T *ptr();

        template<typename T>
        const T *ptr() const;

        void reshape(const std::vector<int32_t> &dims);

        std::shared_ptr<base::Buffer> get_buffer() const;

        size_t size() const;

        size_t byte_size() const;

        int32_t dims_size() const;

        base::DataType data_type() const;

        int32_t get_dim(int32_t idx) const;

        const std::vector<int32_t> &dims() const;

        std::vector<size_t> strides() const;

        bool assign(std::shared_ptr<base::Buffer> buffer);

        void reset(base::DataType data_type, const std::vector<int32_t> &dims);

        void set_device_type(base::DeviceType device_type) const;

        base::DeviceType device_type() const;

        bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc = false);

        template<typename T>
        T *ptr(int64_t index);

        template<typename T>
        const T *ptr(int64_t index) const;

        template<typename T>
        T &index(int64_t offset);

        template<typename T>
        const T &index(int64_t offset) const;

        tensor::Tensor clone() const;

    private:
        size_t size_ = 0;//张量中数据的个数
        std::vector<int32_t> dims_;
        std::shared_ptr<base::Buffer> buffer_;//Buffer类用来管理用分配器申请到的内存资源
        base::DataType data_type_ = base::DataType::kDataTypeUnknown;
    };

    template<typename T>
    T &Tensor::index(int64_t offset) {
        CHECK_GE(offset, 0);
        CHECK_LT(offset, this->size());
        T &val = *(reinterpret_cast<T *>(buffer_->ptr()) + offset);
        return val;
    }

    template<typename T>
    const T &Tensor::index(int64_t offset) const {
        CHECK_GE(offset, 0);
        CHECK_LT(offset, this->size());
        const T &val = *(reinterpret_cast<T *>(buffer_->ptr()) + offset);
        return val;
    }
    //template<typename T>
    //这是一段模板函数，意味着你可以通过不同的 T 类型，获得不同类型的数据指针：
    //tensor.ptr<float>() → 返回 const float*
    //tensor.ptr<int32_t>() → 返回 const int32_t*
    //这样就避免了写很多重复函数。
    template<typename T>
    const T *Tensor::ptr() const {//返回当前张量 数据缓冲区的只读指针 （类型为const T*）
        if (!buffer_) {
            return nullptr;//如果这个缓冲区是空的 即张量还没分配内存 返回空指针
        }

        //buffer_->ptr()返回的是 void* 指针，表示“未知类型的内存起始地址”。
        // reinterpret_cast<T*>(...)  将把 void* 强转成 T*，告诉编译器：我们要把它当成 T 类型数组来访问。
        //因为函数是 const 的，只允许读数据，所以这里使用 const_cast 添加 const 限制，最终返回的是 const T*。
        //const_cast 可以让你从 const 指针或引用中移除 const，从而修改原本不能修改的变量（
        return const_cast<const T *>(reinterpret_cast<T *>(buffer_->ptr()));
    }

    template<typename T>
    T *Tensor::ptr() {
        if (!buffer_) {
            return nullptr;
        }
        return reinterpret_cast<T *>(buffer_->ptr());
    }

    template<typename T>
    T *Tensor::ptr(int64_t index) {
        CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
                        << "The data area buffer of this tensor is empty or it points to a null pointer.";
        return const_cast<T *>(reinterpret_cast<const T *>(buffer_->ptr())) + index;
        //这是因为T*的步长和void*的步长是不一致的，
        //比如我要访问ptr+1的位置，如果是直接将void*+1那么指针位置只会往后移动一个位置，如果是float*，那么指针的位置是往后移动4个位置的。
    }

    template<typename T>
    const T *Tensor::ptr(int64_t index) const {
        CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
                        << "The data area buffer of this tensor is empty or it points to a null pointer.";
        return reinterpret_cast<const T *>(buffer_->ptr()) + index;
    }
}  // namespace tensor
#endif  // KUIPER_INCLUDE_TENSOR_TENSOR_H_
