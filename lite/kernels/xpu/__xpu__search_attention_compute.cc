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

#include "lite/kernels/xpu/__xpu__search_attention_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/backends/xpu/debug.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUSearchAttentionCompute::PrepareForRun() {
  //auto& param = this->Param<param_t>();

  void* offset_xpu_ptr = nullptr;
  xpu_malloc(&offset_xpu_ptr, 64 * sizeof(int));
  offset_xpu_guard_.reset(offset_xpu_ptr);

  void* pad_begin_xpu_ptr = nullptr;
  xpu_malloc(&pad_begin_xpu_ptr, 64 * sizeof(int));
  pad_begin_xpu_guard_.reset(pad_begin_xpu_ptr);

  void* w_max_xpu_ptr = nullptr;
  xpu_malloc(&w_max_xpu_ptr, 8 * sizeof(float));
  w_max_xpu_guard_.reset(w_max_xpu_ptr);

  void* buffer_at_l3_ptr = nullptr;
  xpu_malloc(&buffer_at_l3_ptr, 5 * l3_slot_size * sizeof(float));
  buffer_at_l3_guard_.reset(buffer_at_l3_ptr);

  void* buffer_at_gm_ptr = nullptr;
  xpu_malloc(&buffer_at_gm_ptr, 5 * gm_slot_size * sizeof(float));
  buffer_at_gm_guard_.reset(buffer_at_gm_ptr);
}

void XPUSearchAttentionCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* X = param.X;
  auto* W = param.W;
  auto* b = param.b;
  float W_max = param.W_max;
  float alpha0 = param.alpha0;
  float alpha1 = param.alpha1;
  float mask = param.mask;

  //paddle::lite::xpu::dump_xpu_mem(X->data<float>(),
      //X->numel(),
      //"attention bottom", 128, 16);

  const int16_t* w_data = W->data<int16_t>();
  const float* b_data = b->data<float>();

  int batch = X->lod()[0].size() - 1;
  int dim0 = X->dims()[0];
  int dim1 = X->dims()[1];
  const auto offset = X->lod()[0];
  int max_seq = 0;

    //auto* top = ctx.Output<LoDTensor>("Out");
    auto* top = param.Out;
    LoD top_lod;
    top_lod.push_back(X->lod()[0]);
    top->set_lod(top_lod);
    top->Resize({dim0, dim1});
    auto* top_data = top->mutable_data<float>(TARGET(kXPU));

  std::unique_ptr<int[]> offset_cpu(new int[offset.size()]);
  std::unique_ptr<int[]> pad_begin_cpu(new int[batch]);
  float maxs_cpu[8] = {0.0f, 0.0f, 0.0f, 0.0f, W_max, 0.0f, 0.0f, 0.0f};
  for (int i = 0; i < batch; ++i) {
      offset_cpu[i] = offset[i]; // type of offset is int64, not supported by xpu
      pad_begin_cpu[i] = offset[i + 1] - offset[i];
      if (offset[i + 1] - offset[i] > max_seq) {
          max_seq = offset[i + 1] - offset[i];
      }
  }
  offset_cpu[batch] = offset[batch];

  xpu_memcpy(offset_xpu_guard_.get(), offset_cpu.get(),
      offset.size() * sizeof(int), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  xpu_memcpy(pad_begin_xpu_guard_.get(), pad_begin_cpu.get(),
      batch * sizeof(int), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  xpu_memcpy(w_max_xpu_guard_.get(), &maxs_cpu[0],
      8 * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);

    int* offset_xpu = (int*)offset_xpu_guard_.get();
    int* pad_begin_xpu = (int*)pad_begin_xpu_guard_.get();
    float* maxs_xpu = (float*)w_max_xpu_guard_.get();
    float* buffer_at_l3 = (float*)buffer_at_l3_guard_.get();
    float* buffer_at_gm = (float*)buffer_at_gm_guard_.get();

    // when use l3, max_seq <= 128:
    // group_padding:                   batch * max_seq * dim1;           at (slot0, slot1)
    // seq_fc:                          batch * max_seq * dim1;           at (slot2, slot3)
    // batchgemm0:                      batch * max_seq * max_seq;        at slot4
    // attention_padding_mask:          batch * max_seq * max_seq;        at slot3
    // seq_softmax:                     batch * max_seq * max_seq;        at slot4
    // batchgemm1:                      batch * max_seq * dim1;           at (slot2, slot3)
    float* group_padding_output = buffer_at_l3;
    float* seq_fc_output = buffer_at_l3 + 2 * l3_slot_size;
    float* batchgemm0_output = buffer_at_l3 + 4 * l3_slot_size;
    float* attention_output = buffer_at_l3 + 3 * l3_slot_size;
    float* seq_softmax_output = buffer_at_l3 + 4 * l3_slot_size;
    float* batchgemm1_output = buffer_at_l3 + 2 * l3_slot_size;

    if (max_seq > 128) {
        group_padding_output = buffer_at_gm;
        seq_fc_output = buffer_at_gm + 1 * gm_slot_size;
        batchgemm0_output = buffer_at_gm + 2 * gm_slot_size;
        attention_output = buffer_at_gm + 1 * gm_slot_size;
        seq_softmax_output = buffer_at_gm + 3 * gm_slot_size;
        batchgemm1_output = buffer_at_gm + 4 * gm_slot_size;
    }


    const auto* bottom_data = X->data<float>();
    int r = xdnn::search_sequence_pad_depad(ctx.GetRawContext(),
            const_cast<float*>(bottom_data), group_padding_output, offset_xpu, max_seq, batch, dim1, 0); // is_depad = 0
    //PADDLE_ENFORCE(r == xpu::Error_t::SUCCESS, "XPU kernel error!");
    (void)r;
    //{
      ////cwndmiao debug
      //size_t expected_len = batch * max_seq * dim1;
      //paddle::lite::xpu::dump_xpu_mem(group_padding_output,
          //expected_len,
          //"group padding", 128, 16);
    //}

    // do-findmax
    r = xdnn::findmax<float>(ctx.GetRawContext(), group_padding_output, batch * max_seq * dim1, maxs_xpu);
    //PADDLE_ENFORCE(r == xpu::Error_t::SUCCESS, "XPU kernel error!");
    //{
      ////cwndmiao debug
      //size_t expected_len = 8;
      //paddle::lite::xpu::dump_xpu_mem(maxs_xpu,
          //expected_len,
          //"findmax");
    //}

    r = xdnn::gemm_int16_maxptr<float, int16_t, float>(ctx.GetRawContext(),
            false, true,                        //trans_a, trans_b
            batch * max_seq, dim1, dim1,        //m, n, k
            1.0f, group_padding_output, dim1,    //alpha, data_a, lda
            w_data, dim1, 0.0f,                 //data_b, ldb, beta
            seq_fc_output, dim1, b_data,          // data_c, ldc, bias
            xdnn::Activation_t::LINEAR,
            maxs_xpu, maxs_xpu + 4, nullptr); //max_a, max_b, max_c
    //PADDLE_ENFORCE(r == xpu::Error_t::SUCCESS, "XPU kernel error!");
    //{
      //size_t expected_len = dim1 * dim1;
      //paddle::lite::xpu::dump_xpu_mem(w_data,
          //expected_len,
          //"w_data", 32, 32);
    //}
    //{
      ////cwndmiao debug
      //size_t expected_len = batch * max_seq * dim1;
      //paddle::lite::xpu::dump_xpu_mem(seq_fc_output,
          //expected_len,
          //"search_seq_fc", 128, 32);
    //}

    r = xdnn::search_aligned_mat_mul(ctx.GetRawContext(),
            0, 1, batch, max_seq, max_seq, dim1, alpha0,
            group_padding_output, dim1, seq_fc_output, dim1,
            batchgemm0_output, max_seq);
    //PADDLE_ENFORCE(r == xpu::Error_t::SUCCESS, "XPU kernel error!");

    r = xdnn::search_pad_mask(ctx.GetRawContext(),
            batchgemm0_output, attention_output,
            pad_begin_xpu, batch, max_seq, max_seq, batch, mask);
    //PADDLE_ENFORCE(r == xpu::Error_t::SUCCESS, "XPU kernel error!");

    r = xdnn::softmax2d_forward(ctx.GetRawContext(),
            attention_output, seq_softmax_output, batch * max_seq, max_seq, true);
    //PADDLE_ENFORCE(r == xpu::Error_t::SUCCESS, "XPU kernel error!");

    r = xdnn::search_aligned_mat_mul(ctx.GetRawContext(),
            0, 0, batch, max_seq, dim1, max_seq, alpha1,
            seq_softmax_output, max_seq, group_padding_output, dim1,
            batchgemm1_output, dim1);
    //PADDLE_ENFORCE(r == xpu::Error_t::SUCCESS, "XPU kernel error!");

    r = xdnn::search_sequence_pad_depad(ctx.GetRawContext(),
            top_data, batchgemm1_output, offset_xpu, max_seq, batch, dim1, 1); // is_depad = 1
    //PADDLE_ENFORCE(r == xpu::Error_t::SUCCESS, "XPU kernel error!");
    //
  //paddle::lite::xpu::dump_xpu_mem(top->data<float>(),
      //top->numel(),
      //"attention top", 128, 16);

}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__search_attention,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUSearchAttentionCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
