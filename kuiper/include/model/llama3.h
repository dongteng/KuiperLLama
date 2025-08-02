#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_

#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"

namespace model {

    struct LLama2Layers {
        std::shared_ptr<op::Layer> add_layer_;//残差连接中的加法层（Residual Add）。
        std::shared_ptr<op::Layer> rope_layer_;//注意力机制中引入位置编码信息
        std::shared_ptr<op::Layer> swiglu_layer_;
        std::shared_ptr<op::Layer> mha_layer_;

        //多权重矩阵层，为什么用vector，每个头可能有独立的权重层，也可以是分片并行
        std::vector<std::shared_ptr<op::Layer>> wq_layers_;
        std::vector<std::shared_ptr<op::Layer>> wk_layers_;
        std::vector<std::shared_ptr<op::Layer>> wv_layers_;
        std::vector<std::shared_ptr<op::Layer>> wo_layers_;

        //MLP和归一化层
        std::vector<std::shared_ptr<op::Layer>> w1_layers_;
        std::vector<std::shared_ptr<op::Layer>> w2_layers_;
        std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
        std::vector<std::shared_ptr<op::Layer>> w3_layers_;
        //分类输出层，用于最终预测（可选）
        std::shared_ptr<op::Layer> cls_layer_;

        std::shared_ptr<op::Layer> embedding_layer_;

        void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
    };

    class LLama2Model : public Model {
    public:
        explicit LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                             std::string model_path, bool is_quant_model);

        base::Status init(base::DeviceType device_type) override;

        base::Status predict(const tensor::Tensor &input, const tensor::Tensor &pos_tensor,
                             bool is_prompt, int &next) const override;

        base::Status forward(const tensor::Tensor &input, const tensor::Tensor &pos_tensor,
                             int &next) const override;

        op::EmbeddingOutput embedding(const std::vector<int> &tokens) const override;

    private:
        void init_mem() override;

        base::Status create_layers() override;

        void create_param_layers() override;

        void create_nonparam_layers() override;

        void create_param_quant_layers() override;

        void attention_mha(int32_t layer_idx, const tensor::Tensor &pos_tensor) const;

        void attention_rms(int32_t layer_idx, const tensor::Tensor &input) const;

        void feed_forward(int32_t layer_idx, const tensor::Tensor &input) const;

        void attention_qkv(int32_t layer_idx, const tensor::Tensor &pos_tensor) const;

        void cls_logits(const tensor::Tensor &input) const;

        int32_t post_processing(const tensor::Tensor &pos, bool is_prompt) const override;

    private:
        std::shared_ptr<kernel::CudaConfig> cuda_config_;
        std::unique_ptr<LLama2Layers> llama_layers_;
    };
}  // namespace model

#endif