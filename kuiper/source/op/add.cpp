#include "op/add.h"
#include "kernels/kernels_interface.h"

namespace op {
    VecAddLayer::VecAddLayer(base::DeviceType device_type)
            : Layer(device_type, LayerType::kLayerAdd, "Add") {
        reset_input_size(2);
        reset_output_size(1);
    }

    base::Status VecAddLayer::check() const {
        tensor::Tensor input1 = this->get_input(0);
        tensor::Tensor input2 = this->get_input(1);
        int32_t size = input1.size();//使用 int32_t 是为了在跨平台、跨模块、跨语言交互时，让下标 idx 的类型保持稳定、固定为 32 位有符号整数，避免潜在的类型不一致错误
        base::Status status;
        status = check_tensor_with_dim(input1, device_type_, data_type_, size);
        if (!status) {
            LOG(ERROR) << "The input tensor 1 error in the add layer.";
            return status;
        }

        status = check_tensor_with_dim(input2, device_type_, data_type_, size);
        if (!status) {
            LOG(ERROR) << "The input tensor 2 error in the add layer.";
            return status;
        }

        status = check_tensor_with_dim(get_output(0), device_type_, data_type_, size);
        if (!status) {
            LOG(ERROR) << "The output tensor error in the add layer.";
            return status;
        }
        return base::error::Success();
    }

    base::Status VecAddLayer::forward() {
        auto status = this->check();
        if (!status) {
            return status;
        }
        auto input1 = this->get_input(0);//这里get_input函数 定义的是引用函数
        auto input2 = this->get_input(1);
        auto output = this->get_output(0);
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            CHECK(cuda_config_ != nullptr);
        }
        kernel::get_add_kernel(device_type_)(input1, input2, output,
                                             cuda_config_ ? cuda_config_->stream : nullptr);
        return base::error::Success();
    }

}  // namespace op