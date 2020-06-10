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

#include "lite/kernels/xpu/var_conv_2d_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void VarConv2DCompute::PrepareForRun() {
  offset_x_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(64 * sizeof(int));
  offset_y_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(64 * sizeof(int));
  offset_x_cpu.reset(new int[64]);
  offset_y_cpu.reset(new int[64]);
}

void VarConv2DCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* bottom = param.X;
  auto* w = param.W;
  auto* top = param.Out;
  //auto* col = ctx.Output<LoDTensor>("Col");

  int output_channel = param.output_channel;
  int input_channel = param.input_channel;
  int kernel_h = param.kernel_h;
  int kernel_w = param.kernel_w;
  int stride_h = param.stride_h;
  int stride_w = param.stride_w;
  float max_w = param.max_w;
  bool fuse_relu = param.fuse_relu;
  bool float_to_fix = param.float_to_fix;

  CHECK(float_to_fix) << "W should be fixed point";

  xdnn::Activation_t act = xdnn::Activation_t::LINEAR;
  if (fuse_relu) {
    act = xdnn::Activation_t::RELU;
  }

  int batch = bottom->lod()[0].size() - 1;
  const auto& offset_x = bottom->lod()[2];
  const auto& offset_y = bottom->lod()[1];
  std::vector<size_t> top_offset;
  int top_size = 0;
  top_offset.push_back(top_size);
  for (int b = 0; b < batch; ++b) {
    int width = offset_x[b + 1] - offset_x[b];
    int height = offset_y[b + 1] - offset_y[b];
    int top_im_x = 0;
    int top_im_y = 0;
    if (width != 0) {
      top_im_x = (width - 1) / stride_w + 1;
    }
    if (height != 0) {
      top_im_y = (height - 1) / stride_h + 1;
    }
    int top_im_size = top_im_y * top_im_x;
    top_size += output_channel * top_im_size;
    top_offset.push_back(top_size);
  }

  LoD top_lod;
  top_lod.push_back(top_offset);
  top_lod.push_back(bottom->lod()[1]);
  top_lod.push_back(bottom->lod()[2]);

  top->set_lod(top_lod);
  std::vector<int64_t> top_dims_vec{top_size};
  top_dims_vec.push_back(1);
  top->Resize(top_dims_vec);
  auto* top_data = top->mutable_data<float>(TARGET(kXPU));

  auto* bottom_data = bottom->data<float>();
  auto* w_data = w->data<int16_t>();

  // TODO(chenrong06) col_data is only used in traning maybe?
  //auto* col_data = col->data<T>();

  //auto& dev_ctx = ctx.template device_context<DeviceContext>();

  int* offset_x_xpu = nullptr;
  int* offset_y_xpu = nullptr;
  //offset_x_xpu = (int*) xpu::alloc_workspace(dev_ctx.x_context(), (batch + 1) * sizeof(int));
  //offset_y_xpu = (int*) xpu::alloc_workspace(dev_ctx.x_context(), (batch + 1) * sizeof(int));
  //PADDLE_ENFORCE(offset_x_xpu != nullptr, "Fail to alloc L3");
  //PADDLE_ENFORCE(offset_y_xpu != nullptr, "Fail to alloc L3");
  offset_x_xpu = (int*)offset_x_xpu_guard_->addr_;
  offset_y_xpu = (int*)offset_y_xpu_guard_->addr_;
  //int* offset_x_cpu = (int*)malloc((batch + 1) * sizeof(int));
  //int* offset_y_cpu = (int*)malloc((batch + 1) * sizeof(int));
  //std::unique_ptr<int[]> offset_x_cpu(new int[batch + 1]);
  //std::unique_ptr<int[]> offset_y_cpu(new int[batch + 1]);
  for (int i = 0; i < (batch + 1); ++i) {
      offset_x_cpu[i] = offset_x[i];
      offset_y_cpu[i] = offset_y[i];
  }
  //memory::Copy(
          //boost::get<platform::XPUPlace>(dev_ctx.GetPlace()),
          //(void*)offset_x_xpu,
          //platform::CPUPlace(), (void*)offset_x_cpu,
          //(batch + 1) * sizeof(int));
  //memory::Copy(
          //boost::get<platform::XPUPlace>(dev_ctx.GetPlace()),
          //(void*)offset_y_xpu,
          //platform::CPUPlace(), (void*)offset_y_cpu,
          //(batch + 1) * sizeof(int));
  xpu_memcpy(offset_x_xpu,
      offset_x_cpu.get(),
      (batch + 1) * sizeof(int),
      XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  xpu_memcpy(offset_y_xpu,
      offset_y_cpu.get(),
      (batch + 1) * sizeof(int),
      XPUMemcpyKind::XPU_HOST_TO_DEVICE);

  int ret = xdnn::search_varconv<float, int16_t>(ctx.GetRawContext(),
          batch, input_channel, output_channel, kernel_h, kernel_w, stride_h, stride_w,
          bottom_data, w_data, offset_x_xpu, offset_y_xpu,
          top_data, max_w, act);
  //PADDLE_ENFORCE(ret == xpu::Error_t::SUCCESS, "XPU kernel error!");
  CHECK_EQ(ret, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(var_conv_2d,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::VarConv2DCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Col", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
