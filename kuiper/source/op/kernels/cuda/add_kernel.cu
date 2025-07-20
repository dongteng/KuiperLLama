#include "add_kernel.cuh"

namespace kernel {
    __global__ void add_kernel_cu_fp32(int32_t size, const float *in1, const float *in2, float *out) {
        int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
        if (tid >= size) {
            return;
        }
        float in_val1 = in1[tid];
        float in_val2 = in2[tid];
        out[tid] = in_val1 + in_val2;
    }

    void add_kernel_cu(const tensor::Tensor &input1, const tensor::Tensor &input2,
                       const tensor::Tensor &output, void *stream) {//stream：CUDA 流指针，如果不为空就用这个流来调度 kernel，否则用默认流。
        CHECK_EQ(input1.is_empty(), false);
        CHECK_EQ(input2.is_empty(), false);
        CHECK_EQ(output.is_empty(), false);//确保你已经准备好 output 的空间，GPU 才能往里面写。否则 kernel 写入就可能访问非法内存导致崩溃。
        int32_t size = static_cast<int32_t>(input1.size());
        CHECK_EQ(size, input2.size());
        CHECK_EQ(size, output.size());
        int32_t thread_num = 512;
        int32_t block_num = (size + thread_num - 1) / thread_num;
        if (stream) {
            cudaStream_t stream_ = static_cast<CUstream_st *>(stream);
            add_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
                    size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float *>(output.ptr<float>()));
        } else {
            //没有指定stream 默认流 ，执行是同步顺序调度的，简单安全 效率可能不如stream高
            add_kernel_cu_fp32<<<block_num, thread_num>>>(size, input1.ptr<float>(), input2.ptr<float>(),
                                                          const_cast<float *>(output.ptr<float>()));
        }
    }
}  // namespace kernel
