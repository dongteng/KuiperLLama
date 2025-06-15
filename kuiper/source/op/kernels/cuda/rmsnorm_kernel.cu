#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"

namespace kernel {
    template<int32_t BLOCK_DIM> //template<int32_t BLOCK_DIM>：这是一个模板函数，BLOCK_DIM 表示每个 CUDA 线程块（block）的线程数（如 128、256 等）。
    static __global__ void row_rmsnorm_f32(float *in, float *wei, float *out, int size, float eps) {//cuda核函数
        //size是输入向量长度


        const int tid = threadIdx.x;//threadIdx.x是cuda线程编号


        //假设size =10  那么 pack_num为2  pack_off= 2*4 =8
        constexpr int pack_size = 4; //onstexpr 表示“编译期常量”
        const int pack_num = size / pack_size;
        const int pack_off = pack_size * pack_num; //这两行代码的作用是为了实现SIMD 向量化计算（一次处理多个 float），比如通过 float4 一次处理 4 个

        float sum = 0.0f; //定义一个局部变量sum  每个县城独立累加它所负责的元素的平方和
        float4 *in_pack = reinterpret_cast<float4 *>(in);//把 in 这个 float* 类型的指针（指向普通 float 数组）强行解释成 float4* 类型的指针。
        //例float in[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};  那么 reinterpret_cast<float4 *>(in) 把它“重新解释”成每 4 个 float 为一组的结构：
//        in_pack[0] --> float4(1.0f, 2.0f, 3.0f, 4.0f)
//        in_pack[1] --> float4(5.0f, 6.0f, 7.0f, 8.0f)  这叫矢量化读取或SIMD读取




        //pack_num表示有多少个float4元素
        //blockDim.x 表示每个线程块中线程的数量
        //每个线程从tid开始，以blockDim.x为步长遍历，例如 线程0处理 i=0, i=128, i=256...；线程 1 处理 i=1, i=129, i=257...
        for (int i = tid; i < pack_num; i += blockDim.x) {
            float4 in_float4 = *(in_pack + i);//取一个float4数据块
            sum += in_float4.x * in_float4.x;
            sum += in_float4.y * in_float4.y;
            sum += in_float4.z * in_float4.z;
            sum += in_float4.w * in_float4.w;
        }

        for (int i = pack_off + tid; i < size; i += blockDim.x) {//处理剩下的不是4的倍数的尾部数据
            sum += in[i] * in[i];
        }

        //使用CUB做线程块内部归约（平方和）
        using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;//定义一个用于 块归约（block-wide reduction）的别名 BlockReduce。
        __shared__ typename BlockReduce::TempStorage temp;//temp 是 CUB 要求的 共享内存临时存储空间。
        __shared__ float shared_val;//shared_val 是 block 内所有线程归约得到的最终平方和。
        sum = BlockReduce(temp).Sum(sum);//这行相当于 sum = ∑（当前线程块中所有线程的 sum）
        if (threadIdx.x == 0) { //线程块中只有一个线程（编号 0）将 sum 保存到共享内存变量 shared_val 中。
            shared_val = sum;
        }
        __syncthreads();//其他线程通过 __syncthreads() 等待并读取这个值。只有等所有线程都执行到这里了，才能继续往下执行
        sum = shared_val;//所有线程执行，安全读取共享内存中保存的规约值 可以保证所有线程拿到的是一样且有效的结果


        //sum / static_cast<float>(size)是得到 平方和 sum 除以元素总数 size，
        // rsqrtf CUDA 内置函数，计算 1 / sqrt(x)，即倒数平方根，速度快于先算开根号再取倒数。
        const float scale = rsqrtf(sum / static_cast<float>(size) + eps);//

        float4 *wei_pack = reinterpret_cast<float4 *>(wei);
        float4 *out_pack = reinterpret_cast<float4 *>(out);
        for (int i = tid; i < pack_num; i += blockDim.x) {
            float4 in_float4 = *(in_pack + i);
            float4 wei_float4 = *(wei_pack + i);
            *(out_pack + i) =
                    make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                                scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
        }

        for (int i = pack_off + tid; i < size; i += blockDim.x) {
            out[i] = wei[i] * in[i] * scale;
        }
    }

    void rmsnorm_kernel_cu(const tensor::Tensor &input, const tensor::Tensor &weight,
                           const tensor::Tensor &output, void *stream) {
        CHECK(!input.is_empty());
        CHECK(!weight.is_empty());
        CHECK(!output.is_empty());

        CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
              weight.device_type() == base::DeviceType::kDeviceCUDA &&
              output.device_type() == base::DeviceType::kDeviceCUDA);

#ifdef QWEN2_SUPPORT
        const float eps = 1e-6f;
#else
        const float eps = 1e-5f;
#endif
        const int32_t size = static_cast<int32_t>(input.size());
        float *in_ptr = const_cast<float *>(input.ptr<float>());
        float *wei_ptr = const_cast<float *>(weight.ptr<float>());
        float *out_ptr = const_cast<float *>(output.ptr<float>());
        constexpr int threads_num = 128;
        if (stream) {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);//行代码的作用是将上层框架传递的通用流句柄 stream 安全地转换为 CUDA 原生流类型 cudaStream_t，以便后续 CUDA 核函数能够在正确的流上异步执。
            row_rmsnorm_f32<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
        } else {
            row_rmsnorm_f32<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
        }
    }
}  // namespace kernel