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
      ->AsInput();
    auto* search_seq_fc_b = VarNode("search_seq_fc_b")
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

}  // namespace fusion

class XPUMMDNNFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    fusion::XPUMMDNNSearchAttentionFuser block0_fuser;
    block0_fuser(graph.get());
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
    .BindKernel("conv2d");
