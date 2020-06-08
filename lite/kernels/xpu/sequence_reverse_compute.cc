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

#include "lite/kernels/xpu/sequence_reverse_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

//namespace {

//template <paddle::lite_api::PrecisionType PType>
//struct PrecisionType2BuiltinType;

//template <>
//struct PrecisionType2BuiltinType<PRECISION(kFloat)> { typedef float type; };

//template <>
//struct PrecisionType2BuiltinType<PRECISION(kInt64)> { typedef int64_t type; };

//}

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void XPUSequenceReverseCompute<T, PType>::PrepareForRun() {
  void* lod_xpu_ptr = nullptr;
  xpu_malloc(&lod_xpu_ptr, 64 * sizeof(int));
  lod_xpu_guard_.reset(lod_xpu_ptr);
}

template <typename T, PrecisionType PType>
void XPUSequenceReverseCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* x = param.X;
  auto* y = param.Out;

  auto lod = x->lod()[0];
  size_t limit = x->numel();
  size_t ele_cnt_in_4_byte = limit / x->dims()[0];
  auto* x_data = x->template data<T>();
  auto* y_data = y->template mutable_data<T>(TARGET(kXPU));

  if (std::is_same<T, uint8_t>::value) {
      //PADDLE_ENFORCE(ele_cnt_in_4_byte % 4 == 0, "ele_cnt_in_4_byte % 4 != 0");
      ele_cnt_in_4_byte /= 4;
  } else if (std::is_same<T, int>::value) {
      // remain the same
  } else if (std::is_same<T, int64_t>::value) {
      ele_cnt_in_4_byte *= 2;
  } else if (std::is_same<T, float>::value) {
      // remain the same
  } else if (std::is_same<T, double>::value) {
      ele_cnt_in_4_byte *= 2;
  }

  int batch_size = lod.size() - 1;

  std::unique_ptr<int[]> lod_cpu(new int[lod.size()]);
  for (size_t i = 0; i < lod.size(); ++i) {
    lod_cpu[i] = lod[i];
  }
  xpu_memcpy(lod_xpu_guard_.get(), lod_cpu.get(), lod.size() * sizeof(int), XPUMemcpyKind::XPU_HOST_TO_DEVICE);

  xdnn::sequence_reverse(ctx.GetRawContext(),
      batch_size, (const int*)lod_xpu_guard_.get(),
      ele_cnt_in_4_byte,
      (const float*)x_data, (float*)y_data);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;
using SequenceReverseFp32 = xpu::XPUSequenceReverseCompute<float, PRECISION(kFloat)>;
using SequenceReverseInt64 = xpu::XPUSequenceReverseCompute<int64_t, PRECISION(kInt64)>;

REGISTER_LITE_KERNEL(sequence_reverse, kXPU, kFloat, kNCHW, SequenceReverseFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(sequence_reverse, kXPU, kInt64, kNCHW, SequenceReverseInt64, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
