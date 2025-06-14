#include "add_kernel.h"
#include <armadillo>//高性能线性代数库 支持向量矩阵运算
#include "base/base.h"

namespace kernel {
    void add_kernel_cpu(const tensor::Tensor &input1, const tensor::Tensor &input2,
                        const tensor::Tensor &output, void *stream) {
        UNUSED(stream);//这行用于避免未使用参数编译警告， 因为在 CPU 模式下，不需要 stream，但接口可能统一保留。
        CHECK_EQ(input1.is_empty(), false);
        CHECK_EQ(input2.is_empty(), false);
        CHECK_EQ(output.is_empty(), false);

        CHECK_EQ(input1.size(), input2.size());
        CHECK_EQ(input1.size(), output.size());

        //使用armadillo包装张量数据
        //各参数解释 arma::fvec(data_ptr, n_elem, copy_aux_mem = false, strict = true);
        //ptr<float>() 是对 ptr() 函数的调用； <float> 指定模板类型参数为 float，也就是说你希望把底层的数据缓冲区解释为 float* 类型；
        //input1.ptr<float>()	是一个模板函数调用，返回 const float*，即只读数据指针
        //const_cast<float *>(...)	把只读指针 const float* 转为可写指针 float*
        //“我用 Tensor 里的数据，构造了一个不复制内存的 Armadillo 一维列向量。”  fvec不存在形状
        arma::fvec input_vec1(const_cast<float *>(input1.ptr<float>()), input1.size(), false, true);
        arma::fvec input_vec2(const_cast<float *>(input2.ptr<float>()), input2.size(), false, true);
        arma::fvec output_vec(const_cast<float *>(output.ptr<float>()), output.size(), false, true);
        output_vec = input_vec1 + input_vec2;
    }

}  // namespace kernel
