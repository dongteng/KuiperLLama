#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <model/config.h>
#include <op/matmul.h>
#include <sys/mman.h>
#include "../source/op/kernels/kernels_interface.h"
#include "base/buffer.h"

TEST(test_load, load_model_config) {
    std::string model_path = "/KuiperLLama/tmp/test.bin";
    int32_t fd = open(model_path.data(), O_RDONLY);
    ASSERT_NE(fd, -1);

//    FILE *file = fopen(model_path.data(), "rb");
    FILE *file = fopen(model_path.c_str(), "rb"); //fopen是C标准库stdio.h提供的一个文件打开函数 ，"rb"：以 二进制只读 模式打开， "r"：以 文本只读 模式打开，"wb"：以 二进制写入 模式打开（会覆盖原文件

    ASSERT_NE(file, nullptr);

    auto config = model::ModelConfig{};
    fread(&config, sizeof(model::ModelConfig), 1, file);//nmemb：数据块的个数
    ASSERT_EQ(config.dim, 16);
    ASSERT_EQ(config.hidden_dim, 128);
    ASSERT_EQ(config.layer_num, 256);
}

TEST(test_load, load_model_weight) {
    std::string model_path = "/KuiperLLama/tmp/test.bin";
    int32_t fd = open(model_path.data(), O_RDONLY); //open提供文件描述符
    ASSERT_NE(fd, -1);

    FILE *file = fopen(model_path.data(), "rb");//提供文件流指针
    ASSERT_NE(file, nullptr);

    auto config = model::ModelConfig{};
    fread(&config, sizeof(model::ModelConfig), 1, file);//用 fread 从文件中读取 一个结构体大小的数据，填充 config。

    fseek(file, 0, SEEK_END);//`SEEK_END`：以文件末尾为基准
    auto file_size = ftell(file);//调用 ftell 获取当前文件指针位置。

    // `nullptr`：由系统决定映射内存的起始地址。
    //  - `file_size`：映射区域的大小（整个文件）。
    //  - `PROT_READ`：映射内存区域是**只读**的。
    //  - `MAP_PRIVATE`：映射区域是**当前进程私有的**，修改不会影响文件。
    //  - `fd`：文件描述符。
    //  - `0`：从文件头开始映射。
    void *data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    float *weight_data =
            reinterpret_cast<float *>(static_cast<int8_t *>(data) + sizeof(model::ModelConfig));

    for (int i = 0; i < config.dim * config.hidden_dim; ++i) {
        ASSERT_EQ(*(weight_data + i), float(i));
    }
}

TEST(test_load, create_matmul) {
    std::string model_path = "/KuiperLLama/tmp/test.bin";
    int32_t fd = open(model_path.data(), O_RDONLY);//O_RDONLY代表只读 不能写入
    ASSERT_NE(fd, -1);

    FILE *file = fopen(model_path.data(), "rb");
    ASSERT_NE(file, nullptr);

    auto config = model::ModelConfig{};
    fread(&config, sizeof(model::ModelConfig), 1, file);//调用 C 语言的 fread 函数，从文件 file 中读取数据。

    fseek(file, 0, SEEK_END);//调用 fseek 函数，把文件指针移动到文件末尾。 0是偏移量
    auto file_size = ftell(file);//ftell(file) 返回当前文件指针的位置。
    //因为之前执行过 fseek(file, 0, SEEK_END)，此时文件指针已经在文件尾部，所以这里返回的是整个文件的字节大小。

    //调用mmap把整个文件映射到内存中
    //nullptr让内核选择映射的内存地址，可以指定，但是这里交给系统分配
    //file_size 是映射区域大小 就是整个文件的字节数。PROT_READ: 这个区域是只读权限。MAP_PRIVATE: 私有映射，修改映射区域不会影响原文件（Copy-on-Write）
    //fd: 文件描述符（注意这里是 fd，不是 file*，一般在用 fopen 后需要 fileno(file) 得到 fd）。
    //0: 文件偏移量，从文件头开始映射。
    //将整个文件直接映射到内存中，返回映射区域的首地址指针data
    void *data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    //data这是一个 void* 指针，指向 文件映射到内存的首地址。由于它是 void*，无法直接进行偏移运算（void* 没有大小）
    //把 data 转成 int8_t*（也可以理解为 char*）。int8_t* 是 字节指针，每次加 +1 偏移一个字节。
    float *weight_data =
            reinterpret_cast<float *>(static_cast<int8_t *>(data) + sizeof(model::ModelConfig));

    for (int i = 0; i < config.dim * config.hidden_dim; ++i) {
        ASSERT_EQ(*(weight_data + i), float(i));
    }
    /**                                  1
     *    1 2 3 4 5 6 ... 1024           1
     *                                   1
     *                                   1
     */
    auto wq = std::make_shared<op::MatmulLayer>(base::DeviceType::kDeviceCPU, config.dim,
                                                config.hidden_dim, false);
    float *in = new float[config.hidden_dim];
    for (int i = 0; i < config.hidden_dim; ++i) {
        in[i] = 1.f;
    }

    float *out = new float[config.dim];
    for (int i = 0; i < config.dim; ++i) {
        out[i] = 0.f;
    }
    tensor::Tensor tensor(base::DataType::kDataTypeFp32, config.hidden_dim, false, nullptr, in);
    tensor.set_device_type(base::DeviceType::kDeviceCPU);

    tensor::Tensor out_tensor(base::DataType::kDataTypeFp32, config.dim, false, nullptr, out);
    out_tensor.set_device_type(base::DeviceType::kDeviceCPU);

    wq->set_input(0, tensor);
    wq->set_output(0, out_tensor);
    wq->set_weight(0, {config.dim, config.hidden_dim}, weight_data, base::DeviceType::kDeviceCPU);
    wq->forward(); // 完成一个计算

    /** python code:
     *  w = np.arange(0,128 * 16).reshape(16, 128)
     *  input = np.ones(128)
     *  out = w@input
     */
    ASSERT_EQ(out[0], 8128);
    ASSERT_EQ(out[1], 24512);
    ASSERT_EQ(out[14], 237504);
    ASSERT_EQ(out[15], 253888);

    delete[] in;
    delete[] out;
}