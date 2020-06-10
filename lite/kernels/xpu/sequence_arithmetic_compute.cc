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

#include "lite/kernels/xpu/sequence_arithmetic_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SequenceArithmeticCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* bottom0 = param.X;
  auto* bottom1 = param.Y;
  auto* top = param.Out;

  int _op_type = param.op_type;

  auto len1 = bottom0->numel();
  auto len2 = bottom1->numel();
  const auto* bottom_data0 = bottom0->data<float>();
  const auto* bottom_data1 = bottom1->data<float>();
  auto* top_data = top->mutable_data<float>(TARGET(kXPU));

  //{
    //float debug_data0[5];
    //xpu_memcpy(&debug_data0[0], bottom_data0 + len1 - 5, 5 * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    //printf("cwndmiao debug SequenceArithmeticCompute data0 %f %f %f %f %f\n",
        //debug_data0[0], debug_data0[1], debug_data0[2], debug_data0[3], debug_data0[4]);
  //}
  //{
    //float debug_data1[5];
    //xpu_memcpy(&debug_data1[0], bottom_data1 + len2 - 5, 5 * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    //printf("cwndmiao debug SequenceArithmeticCompute data1 %f %f %f %f %f\n",
        //debug_data1[0], debug_data1[1], debug_data1[2], debug_data1[3], debug_data1[4]);
  //}

  switch (_op_type) {
    case 1:  // addition: top[0] = bottom[0] + bottom[1]
      if (len1 > len2) {
        /*int ret = */xdnn::elementwise_add(ctx.GetRawContext(),
                bottom_data0, bottom_data1, top_data, len2);
        xdnn::memcpy_device(ctx.GetRawContext(),
            (void*)&top_data[len2],
            (void*)&bottom_data0[len2],
            (len1 - len2) * sizeof(float));
      } else {
        /*int ret = */xdnn::elementwise_add(ctx.GetRawContext(),
                bottom_data0, bottom_data1, top_data, len1);
      }
      break;
    case 2:  // substraction: top[0] = bottom[0] - bottom[1]
      if (len1 > len2) {
        /*int ret = */xdnn::elementwise_sub(ctx.GetRawContext(),
                bottom_data0, bottom_data1, top_data, len2);
        xdnn::memcpy_device(ctx.GetRawContext(),
            (void*)&top_data[len2],
            (void*)&bottom_data0[len2],
            (len1 - len2) * sizeof(float));
      } else {
        /*int ret = */xdnn::elementwise_sub(ctx.GetRawContext(),
                bottom_data0, bottom_data1, top_data, len1);
      }
      break;
    case 3:  // multiplication: top[0] = bottom[0] * bottom[1]
      if (len1 > len2) {
        /*int ret = */xdnn::elementwise_mul(ctx.GetRawContext(),
                bottom_data0, bottom_data1, top_data, len2);
        xdnn::memcpy_device(ctx.GetRawContext(),
            (void*)&top_data[len2],
            (void*)&bottom_data0[len2],
            (len1 - len2) * sizeof(float));
      } else {
        /*int ret = */xdnn::elementwise_mul(ctx.GetRawContext(),
                bottom_data0, bottom_data1, top_data, len1);
      }
      break;
    default:
      break;
  }

  //{
    //float debug_top[5];
    //xpu_memcpy(&debug_top[0], top_data + len1 - 5, 5 * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    //printf("cwndmiao debug SequenceArithmeticCompute top %f %f %f %f %f\n",
        //debug_top[0], debug_top[1], debug_top[2], debug_top[3], debug_top[4]);
    //auto dddim = top->dims();
    //printf("dddim %d %d\n", (int)dddim[0], (int)dddim[1]);
  //}
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    sequence_arithmetic,
    kXPU,
    kFloat,
    kNCHW,
    paddle::lite::kernels::xpu::SequenceArithmeticCompute,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    search_seq_arithmetic,
    kXPU,
    kFloat,
    kNCHW,
    paddle::lite::kernels::xpu::SequenceArithmeticCompute,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
