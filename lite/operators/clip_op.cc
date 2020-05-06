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

#include "lite/operators/clip_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ClipOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool ClipOpLite::InferShapeImpl() const {
  param_.Out->Resize(param_.X->dims());
  param_.Out->set_lod(param_.X->lod());
  return true;
}

bool ClipOpLite::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  auto x = op_desc.Input("X").front();
  auto output = op_desc.Output("Out").front();
  param_.X = scope->FindVar(x)->GetMutable<Tensor>();
  param_.Out = scope->FindMutableTensor(output);
  param_.min = op_desc.GetAttr<float>("min");
  param_.max = op_desc.GetAttr<float>("max");
  CHECK(param_.X);
  CHECK(param_.Out);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(clip, paddle::lite::operators::ClipOpLite);
