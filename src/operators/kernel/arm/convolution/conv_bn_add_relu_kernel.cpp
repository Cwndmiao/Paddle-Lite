/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef FUSION_CONVBNADDRELU_OP

#include "operators/kernel/conv_bn_add_relu_kernel.h"
#include <cmath>
#include "operators/kernel/arm/convolution/conv_common.h"
#include "operators/kernel/central-arm-func/conv_arm_func.h"
#include "operators/math/channel_wise.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvBNAddReluKernelCpu<float>::Init(FusionConvBNAddReluParam *param) {
  const Tensor *mean = param->InputMean()->InnerLoDTensor();
  const Tensor *variance = param->InputVariance()->InnerLoDTensor();
  const Tensor *scale = param->InputScale()->InnerLoDTensor();
  const Tensor *bias = param->InputBias()->InnerLoDTensor();
  const float epsilon = param->Epsilon();

  auto mean_ptr = mean->data<float>();
  auto variance_ptr = variance->data<float>();
  auto scale_ptr = scale->data<float>();
  auto bias_ptr = bias->data<float>();

  const int C = mean->numel();
  float inv_std_ptr[C];
  for (int i = 0; i < C; i++) {
    inv_std_ptr[i] =
        1 / static_cast<float>(pow((variance_ptr[i] + epsilon), 0.5));
  }

  auto *new_scale_w = param->CreateNewScale<framework::TensorWrapper>();
  auto *new_bias_w = param->CreateNewBiase<framework::TensorWrapper>();
  LoDTensor *new_scale = new_scale_w->MuteLodTensor();
  LoDTensor *new_bias = new_bias_w->MuteLodTensor();

  auto new_scale_ptr = new_scale->mutable_data<float>({C});
  auto new_bias_ptr = new_bias->mutable_data<float>({C});
  for (int i = 0; i < C; i++) {
    new_scale_ptr[i] = inv_std_ptr[i] * scale_ptr[i];
    new_bias_ptr[i] = bias_ptr[i] - mean_ptr[i] * inv_std_ptr[i] * scale_ptr[i];
  }
  param->SetNewScale(new_scale_w);
  param->SetNewBias(new_bias_w);

  InitBaseConvKernel(param);
  return true;
}

template <>
void ConvBNAddReluKernelCpu<float>::Compute(
    const FusionConvBNAddReluParam &param) {
  switch (param.ExecMode()) {
    case ConvParam::EXEC_DEPTHWISE3x3S1_FLOAT:
    case ConvParam::EXEC_DEPTHWISE3x3S2_FLOAT:
      DepthwiseConv3x3<float, float>(param);
      break;
    case ConvParam::EXEC_DEPTHWISE5x5_FLOAT:
      DepthwiseConv5x5<float, float>(param);
      break;
    case ConvParam::EXEC_WINOGRAD3X3_FLOAT:
      WinogradConv3x3<8, 3>(param);
      break;
    case ConvParam::EXEC_GEMM_FLOAT:
      GemmConv<float, float>(param);
      break;
    default:
      PADDLE_MOBILE_THROW_EXCEPTION("Invalid convolution execute mode %d",
                                    param.ExecMode());
  }
  math::ScaleAddChannelWise<RELU>(
      param.Output()->InnerLoDTensor(), param.NewScale()->InnerLoDTensor(),
      param.NewBias()->InnerLoDTensor(), param.Output()->InnerLoDTensor());
}
template class ConvBNAddReluKernelCpu<float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif