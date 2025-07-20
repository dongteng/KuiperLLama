#include <glog/logging.h>
#include "utils.cuh"

__global__ void test_function_cu(float *cu_arr, int32_t size, float value) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;//计算当前线程全局id   所在块的编号  每个block中线程的数量 当前线程在block内的编号
    if (tid >= size) {
        return;
    }
    cu_arr[tid] = value;
}

void test_function(float *arr, int32_t size, float value) {
    if (!arr) {
        return;
    }
    float *cu_arr = nullptr;
    cudaMalloc(&cu_arr, sizeof(float) * size);
    cudaDeviceSynchronize();
    const cudaError_t err2 = cudaGetLastError();
    test_function_cu<<<1, size>>>(cu_arr, size, value);
    cudaDeviceSynchronize();
    const cudaError_t err = cudaGetLastError();
    CHECK_EQ(err, cudaSuccess);

    cudaMemcpy(arr, cu_arr, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(cu_arr);
}

void set_value_cu(float *arr_cu, int32_t size, float value) {
    int32_t threads_num = 512;//设置每个block启动512个线程
    int32_t block_num = (size + threads_num - 1) / threads_num;//计算启动所需要的block数 向上取整
    cudaDeviceSynchronize();
    const cudaError_t err2 = cudaGetLastError();//获取前一次 CUDA 操作是否有错误。通常这个是为了排查 launch kernel 之前是否出错。
    test_function_cu<<<block_num, threads_num>>>(arr_cu, size, value);//启动 CUDA kernel，对数组进行赋值
    cudaDeviceSynchronize();
    const cudaError_t err = cudaGetLastError();
    CHECK_EQ(err, cudaSuccess);
}
