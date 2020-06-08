// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/xpu_pattern_matcher_high_api.h"
#include "lite/operators/subgraph_op.h"
#include "lite/core/mir/graph_visualize_pass.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class XPUMMDNNSearchAttentionFuser : public FuseBase {
 public:
  XPUMMDNNSearchAttentionFuser() {}

  void BuildPattern() override {
    auto* input = VarNode("input")->AsInput();

    auto* search_group_padding = OpNode("search_group_padding", "search_group_padding");
    auto* out_emb_padding = VarNode("out_emb_padding")
      ->assert_is_op_output("search_group_padding", "Out_emb_padding")
      ->AsIntermediate();
    auto* out_new = VarNode("out_new")
      ->assert_is_op_output("search_group_padding", "Out_new")
      ->AsIntermediate();
    auto* out_padding = VarNode("out_padding")
      ->assert_is_op_output("search_group_padding", "Out_padding")
      ->AsIntermediate();

    auto* search_seq_fc_w = VarNode("search_seq_fc_w")
      ->assert_is_op_input("search_seq_fc", "W")
      ->AsInput();
    auto* search_seq_fc_b = VarNode("search_seq_fc_b")
      ->assert_is_op_input("search_seq_fc", "b")
      ->AsInput();
    auto* search_seq_fc = OpNode("search_seq_fc", "search_seq_fc")
      ->AsIntermediate();
    auto* search_seq_fc_out = VarNode("search_seq_fc_out")
      ->assert_is_op_output("search_seq_fc", "Out")
      ->AsIntermediate();

    auto* search_aligned_mat_mul = OpNode("search_aligned_mat_mul", "search_aligned_mat_mul")
      ->AsIntermediate();
    auto* search_aligned_mat_mul_out = VarNode("search_aligned_mat_mul_out")
      ->assert_is_op_output("search_aligned_mat_mul", "Out")
      ->AsIntermediate();
    auto* search_aligned_mat_mul_a = VarNode("search_aligned_mat_mul_a")
      ->assert_is_op_output("search_aligned_mat_mul", "_a_addr")
      ->AsIntermediate();
    auto* search_aligned_mat_mul_b = VarNode("search_aligned_mat_mul_b")
      ->assert_is_op_output("search_aligned_mat_mul", "_b_addr")
      ->AsIntermediate();
    auto* search_aligned_mat_mul_c = VarNode("search_aligned_mat_mul_c")
      ->assert_is_op_output("search_aligned_mat_mul", "_c_addr")
      ->AsIntermediate();

    auto* search_attention_padding_mask = OpNode("search_attention_padding_mask", "search_attention_padding_mask")
      ->AsIntermediate();
    auto* search_attention_padding_mask_out = VarNode("search_attention_padding_mask_out")
      ->assert_is_op_output("search_attention_padding_mask", "Out")
      ->AsIntermediate();
    auto* search_attention_padding_mask_pad_begin = VarNode("search_attention_padding_mask_pad_begin")
      ->assert_is_op_output("search_attention_padding_mask", "pad_begin")
      ->AsIntermediate();

    auto* search_seq_softmax = OpNode("search_seq_softmax", "search_seq_softmax")
      ->AsIntermediate();
    auto* search_seq_softmax_out = VarNode("search_seq_softmax_out")
      ->assert_is_op_output("search_seq_softmax", "Out")
      ->AsIntermediate();
    auto* search_seq_softmax_out_log = VarNode("search_seq_softmax_out_log")
      ->assert_is_op_output("search_seq_softmax", "Out_log")
      ->AsIntermediate();

    auto* search_aligned_mat_mul_2 = OpNode("search_aligned_mat_mul_2", "search_aligned_mat_mul")
      ->AsIntermediate();
    auto* search_aligned_mat_mul_2_out = VarNode("search_aligned_mat_mul_2_out")
      ->assert_is_op_output("search_aligned_mat_mul", "Out")
      ->AsIntermediate();
    auto* search_aligned_mat_mul_2_a = VarNode("search_aligned_mat_mul_2_a")
      ->assert_is_op_output("search_aligned_mat_mul", "_a_addr")
      ->AsIntermediate();
    auto* search_aligned_mat_mul_2_b = VarNode("search_aligned_mat_mul_2_b")
      ->assert_is_op_output("search_aligned_mat_mul", "_b_addr")
      ->AsIntermediate();
    auto* search_aligned_mat_mul_2_c = VarNode("search_aligned_mat_mul_2_c")
      ->assert_is_op_output("search_aligned_mat_mul", "_c_addr")
      ->AsIntermediate();

    auto* search_seq_depadding = OpNode("search_seq_depadding")
      ->AsIntermediate();
    auto* search_seq_depadding_out = VarNode("search_seq_depadding_out")
      ->AsOutput();

    *input >> *search_group_padding >> *out_emb_padding;
    *search_group_padding >> *out_new;
    *search_group_padding >> *out_padding;

    *search_seq_fc_w >> *search_seq_fc;
    *search_seq_fc_b >> *search_seq_fc;
    *out_emb_padding >> *search_seq_fc;
    *search_seq_fc >> *search_seq_fc_out;

    *search_seq_fc_out >> *search_aligned_mat_mul;
    *out_emb_padding >> *search_aligned_mat_mul;
    *search_aligned_mat_mul >> *search_aligned_mat_mul_out;
    *search_aligned_mat_mul >> *search_aligned_mat_mul_a;
    *search_aligned_mat_mul >> *search_aligned_mat_mul_b;
    *search_aligned_mat_mul >> *search_aligned_mat_mul_c;

    *search_aligned_mat_mul_out >> *search_attention_padding_mask;
    *out_padding >> *search_attention_padding_mask;
    *search_attention_padding_mask >> *search_attention_padding_mask_out;
    *search_attention_padding_mask >> *search_attention_padding_mask_pad_begin;

    *search_attention_padding_mask_out >> *search_seq_softmax;
    *search_seq_softmax >> *search_seq_softmax_out;
    *search_seq_softmax >> *search_seq_softmax_out_log;

    *search_seq_softmax_out >> *search_aligned_mat_mul_2;
    *out_emb_padding >> *search_aligned_mat_mul_2;
    *search_aligned_mat_mul_2 >> *search_aligned_mat_mul_2_out;
    *search_aligned_mat_mul_2 >> *search_aligned_mat_mul_2_a;
    *search_aligned_mat_mul_2 >> *search_aligned_mat_mul_2_b;
    *search_aligned_mat_mul_2 >> *search_aligned_mat_mul_2_c;

    *search_aligned_mat_mul_2_out >> *search_seq_depadding;
    *out_new >> *search_seq_depadding;
    *search_seq_depadding >> *search_seq_depadding_out;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__search_attention");
    op_desc.SetInput("X", {matched.at("input")->arg()->name});
    op_desc.SetInput("W", {matched.at("search_seq_fc_w")->arg()->name});
    op_desc.SetInput("b", {matched.at("search_seq_fc_b")->arg()->name});
    op_desc.SetOutput("Out", {matched.at("search_seq_depadding_out")->arg()->name});
    auto* padding_op_info = matched.at("search_group_padding")->stmt()->op_info();
    op_desc.SetAttr<int>("pad_id", padding_op_info->GetAttr<int>("pad_id"));
    auto* matmul_0_op_info = matched.at("search_aligned_mat_mul")->stmt()->op_info();
    op_desc.SetAttr<float>("alpha0", matmul_0_op_info->GetAttr<float>("alpha"));
    auto* matmul_1_op_info = matched.at("search_aligned_mat_mul_2")->stmt()->op_info();
    op_desc.SetAttr<float>("alpha1", matmul_1_op_info->GetAttr<float>("alpha"));
    auto* mask_op_info = matched.at("search_attention_padding_mask")->stmt()->op_info();
    op_desc.SetAttr<float>("mask", mask_op_info->GetAttr<float>("mask"));

    auto* new_stmt = matched.at("search_group_padding")->stmt();
    auto* scope = new_stmt->op()->scope();
    auto w_name = matched.at("search_seq_fc_w")->arg()->name;
    auto* w_t = scope->FindMutableTensor(w_name);
    auto w_dims = w_t->dims();
    int w_len = w_t->numel();
    float* w_on_host = w_t->mutable_data<float>();

    float max_f = paddle::lite::xpu::math::FindMaxAbs(w_on_host, w_len);
    std::unique_ptr<int16_t[]> w_int16(new int16_t[w_len]);
    paddle::lite::xpu::math::ConvertFP32ToInt16(w_on_host, w_int16.get(), max_f, w_len);
    memcpy(w_on_host, w_int16.get(), w_len * sizeof(int16_t));
    op_desc.SetAttr<float>("W_max", max_f);

    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, scope);
    new_op->SetValidPlaces(new_stmt->op()->valid_places());
    auto kernels = new_op->CreateKernels(new_op->valid_places());
    new_stmt->SetOp(new_op);
    new_stmt->SetKernels(std::move(kernels));

    DirectedLink(matched.at("search_seq_fc_w"), matched.at("search_group_padding"));
    DirectedLink(matched.at("search_seq_fc_b"), matched.at("search_group_padding"));
    IR_OP_VAR_LINK(matched.at("search_group_padding"), matched.at("search_seq_depadding_out"));
  }
};

class Float2Fix {
 public:
  void operator()(SSAGraph* graph) {
    for (auto* node : graph->StmtTopologicalOrder()) {
      CHECK(node->IsStmt());
      auto* op_info = node->stmt()->op_info();
      std::string op_type = op_info->Type();
      static const std::vector<std::string> target_ops{"match_matrix_tensor", "var_conv_2d", "search_fc"};

      if (std::find(target_ops.begin(),
            target_ops.end(), op_type) != target_ops.end()) {
        std::string weight_name = op_info->Input("W").front();
        auto* scope = node->stmt()->op()->scope();
        auto* weight_t = scope->FindMutableTensor(weight_name);
        auto weight_dims = weight_t->dims();
        auto weight_len = weight_t->numel();
        float* weight_on_host = weight_t->mutable_data<float>();
        float max_f =
          paddle::lite::xpu::math::FindMaxAbs(weight_on_host, weight_len);
        std::unique_ptr<int16_t[]> weight_int16(new int16_t[weight_len]);
        paddle::lite::xpu::math::ConvertFP32ToInt16(
            weight_on_host, weight_int16.get(), max_f, weight_len);
        memcpy(weight_on_host,
               weight_int16.get(),
               weight_len * sizeof(int16_t));

        auto update_op_info = *op_info;
        update_op_info.SetAttr<bool>("float_to_fix", true);
        update_op_info.SetAttr<float>("max_w", max_f);
        node->stmt()->ResetOp(update_op_info, graph->valid_places());
        VLOG(3) << "Float2Fix, op_type=" << op_type
          << ", weight_name=" << weight_name;
      } else if (op_type == "search_grnn") {
        auto* scope = node->stmt()->op()->scope();

        std::string wi_name = op_info->Input("Wi").front();
        auto* wi_t = scope->FindMutableTensor(wi_name);
        auto wi_dims = wi_t->dims();
        auto wi_len = wi_t->numel();
        auto wi_stride_len = wi_len / 3;
        float* wi_on_host = wi_t->mutable_data<float>();
        std::unique_ptr<int16_t[]> wi_int16(new int16_t[wi_len]);
        std::vector<float> wi_max(3);
        for (int i = 0; i < 3; ++i) {
          float max_f =
            paddle::lite::xpu::math::FindMaxAbs(wi_on_host + i * wi_stride_len, wi_stride_len);
          paddle::lite::xpu::math::ConvertFP32ToInt16(
              wi_on_host + i * wi_stride_len, wi_int16.get() + i * wi_stride_len, max_f, wi_stride_len);
          wi_max[i] = max_f;
        }
        memcpy(wi_on_host,
               wi_int16.get(),
               wi_len * sizeof(int16_t));

        std::string wh_name = op_info->Input("Wh").front();
        auto* wh_t = scope->FindMutableTensor(wh_name);
        auto wh_dims = wh_t->dims();
        auto wh_len = wh_t->numel();
        auto wh_stride_len = wh_len / 3;
        float* wh_on_host = wh_t->mutable_data<float>();
        std::unique_ptr<int16_t[]> wh_int16(new int16_t[wh_len]);
        std::vector<float> wh_max(3);
        for (int i = 0; i < 3; ++i) {
          float max_f =
            paddle::lite::xpu::math::FindMaxAbs(wh_on_host + i * wh_stride_len, wh_stride_len);
          paddle::lite::xpu::math::ConvertFP32ToInt16(
              wh_on_host + i * wh_stride_len, wh_int16.get() + i * wh_stride_len, max_f, wh_stride_len);
          wh_max[i] = max_f;
        }
        memcpy(wh_on_host,
               wh_int16.get(),
               wh_len * sizeof(int16_t));

        auto update_op_info = *op_info;
        update_op_info.SetAttr<bool>("float_to_fix", true);
        update_op_info.SetAttr<std::vector<float>>("wi_max", wi_max);
        update_op_info.SetAttr<std::vector<float>>("wh_max", wh_max);
        node->stmt()->ResetOp(update_op_info, graph->valid_places());
        VLOG(3) << "Float2Fix, op_type=" << op_type
          << ", wi_name=" << wi_name
          << ", wh_name=" << wh_name;
      }
    }
  }
};

}  // namespace fusion

class XPUMMDNNFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    fusion::XPUMMDNNSearchAttentionFuser block0_fuser;
    block0_fuser(graph.get());
    fusion::Float2Fix float_2_fix;
    float_2_fix(graph.get());
    //fusion::XPUResNetCbamBlock1Fuser block1_fuser;
    //block1_fuser(graph.get());
    //fusion::XPUResNetCbamBlock2Fuser block2_fuser;
    //block2_fuser(graph.get());
    //fusion::XPUResNetCbamFuser resnet_fuser;
    //resnet_fuser(graph.get());

    auto debug_str = Visualize(graph.get());
    printf("debug_str = %s\n", debug_str.c_str());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__mmdnn_fuse_pass,
                  paddle::lite::mir::XPUMMDNNFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__search_attention");
