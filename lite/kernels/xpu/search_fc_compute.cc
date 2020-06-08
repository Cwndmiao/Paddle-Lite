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

#include "lite/kernels/xpu/search_fc_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/backends/xpu/debug.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SearchFcCompute::PrepareForRun() {
  void* maxs_xpu_ptr = nullptr;
  xpu_malloc(&maxs_xpu_ptr, 64 * sizeof(float));
  maxs_xpu_guard_.reset(maxs_xpu_ptr);
}

void SearchFcCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* bottom = param.X;
  auto* w = param.W;
  auto* b = param.b;
  auto* top = param.Out;

  float max_w = param.max_w;
  int out_size = param.out_size;
  bool fuse_relu = param.fuse_relu;
  bool float_to_fix = param.float_to_fix;

  CHECK(float_to_fix) << "W should be fixed point";

  int batch = bottom->dims()[0];

  int _out = w->dims()[0];
  int _in = w->dims()[1];

  xdnn::Activation_t act = xdnn::Activation_t::LINEAR;
  if (fuse_relu) {
    act = xdnn::Activation_t::RELU;
  }

  std::vector<int64_t> top_dims{bottom->dims()[0], out_size};
  top->Resize(top_dims);

  const auto* bottom_data = bottom->data<float>();
  auto* top_data = top->mutable_data<float>(TARGET(kXPU));
  const auto* weights = w->data<int16_t>();
  const float* bias_data = b->data<float>();

  //paddle::lite::xpu::dump_xpu_mem(bottom_data,
      //bottom->numel(),
      //"searchfc bottom", 1, 20);
  //paddle::lite::xpu::dump_xpu_mem(weights,
      //w->numel(),
      //"searchfc weight", 1000, 20);
  //{
    //float debug_data0[5];
    //xpu_memcpy(&debug_data0[0], bottom_data, 5 * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    //printf("SearchFc bottom %f %f %f %f %f\n",
        //debug_data0[0], debug_data0[1], debug_data0[2], debug_data0[3], debug_data0[4]);
  //}
  //{
    //int16_t debug_data0[5];
    //xpu_memcpy(&debug_data0[0], weights, 5 * sizeof(int16_t), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    //printf("SearchFc weight %d %d %d %d %d\n",
        //(int)debug_data0[0], (int)debug_data0[1], (int)debug_data0[2], (int)debug_data0[3], (int)debug_data0[4]);
    //printf("max_w %f\n", max_w);
  //}

  //auto& dev_ctx = ctx.template device_context<DeviceContext>();
  // do-findmax
  //float* maxs_xpu = (float*) xpu::alloc_workspace(dev_ctx.x_context(), 8 * sizeof(float));
  //PADDLE_ENFORCE(maxs_xpu != nullptr, "Fail to alloc L3");
  float* maxs_xpu = (float*)maxs_xpu_guard_.get();
  float maxs_cpu[8] = {0.0f, 0.0f, 0.0f, 0.0f, max_w, 0.0f, 0.0f, 0.0f};
  //memory::Copy(boost::get<platform::XPUPlace>(dev_ctx.GetPlace()),
          //(void*)maxs_xpu, platform::CPUPlace(), (void*)maxs_cpu, 8 * sizeof(float));
  xpu_memcpy(maxs_xpu,
      &maxs_cpu[0],
      8 * sizeof(float),
      XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  int r = xdnn::findmax<float>(ctx.GetRawContext(), bottom_data, batch * _in, maxs_xpu);
  CHECK_EQ(r, 0);
  xpu_memcpy(maxs_cpu, maxs_xpu, 8 * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);

  r = xdnn::gemm_int16_maxptr<float, int16_t, float>(ctx.GetRawContext(),
          false, true, /*trans_a, trans_b*/
          batch, _out, _in, /*m, n, k*/
          1.0f, bottom_data, _in, /*alpha, data_a, lda*/
          weights, _in, 0.0f, /*data_b, ldb, beta*/
          top_data, _out, bias_data, /* data_c, ldc, bias*/
          act, maxs_xpu, maxs_xpu + 4, nullptr /*act, max_a, max_b, max_c*/);
  CHECK_EQ(r, 0);

  //{
    //float debug_data0[5];
    //xpu_memcpy(&debug_data0[0], bias_data, 5 * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    //printf("SearchFc bias %f %f %f %f %f\n",
        //debug_data0[0], debug_data0[1], debug_data0[2], debug_data0[3], debug_data0[4]);
  //}
  //{
    //float debug_data0[5];
    //xpu_memcpy(&debug_data0[0], top_data, 5 * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    //printf("SearchFc top %f %f %f %f %f\n",
        //debug_data0[0], debug_data0[1], debug_data0[2], debug_data0[3], debug_data0[4]);
  //}
  //paddle::lite::xpu::dump_xpu_mem(top_data,
      //top->numel(),
      //"searchfc top", (top->numel() > 100) ? 20 : 1, 20);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(search_fc,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SearchFcCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
