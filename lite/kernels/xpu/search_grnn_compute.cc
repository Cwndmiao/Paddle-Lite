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

#include "lite/kernels/xpu/search_grnn_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/backends/xpu/debug.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SearchGrnnCompute::PrepareForRun() {
  void* offset_xpu_ptr = nullptr;
  xpu_malloc(&offset_xpu_ptr, 64 * sizeof(int));
  offset_xpu_guard_.reset(offset_xpu_ptr);

  void* new_offset_xpu_ptr = nullptr;
  xpu_malloc(&new_offset_xpu_ptr, 64 * sizeof(int));
  new_offset_xpu_guard_.reset(new_offset_xpu_ptr);

  void* maxs_xpu_ptr = nullptr;
  xpu_malloc(&maxs_xpu_ptr, 16 * sizeof(float));
  maxs_xpu_guard_.reset(maxs_xpu_ptr);
}

  void SearchGrnnCompute::xpu_prepare_layout(const operators::SearchGrnnParam& param,
                      const paddle::lite::Tensor* input_blob) const {
    auto* _idx_sorted_by_width = param.idx_sorted_by_width;
    auto* _layout_input = param.layout_input;

    auto _input = input_blob;

    // usually total length
    int dim0 = _input->dims()[0];
    // if it is id only sequence
    int dim1 = 1;

    // if its a embedding like sequence (dim1 would be embedding_size)
    if (_input->dims().size() > 1) {
      dim1 = _input->dims()[1];
    }

    int batch = _input->lod()[0].size() - 1;

    auto& offset = _input->lod()[0];

    _idx_sorted_by_width->Resize({batch});
    _idx_sorted_by_width->mutable_data<int>(TARGET(kXPU));

    paddle::lite::Tensor _width;
    _width.Resize({batch});
    int* width_data = _width.mutable_data<int>();

    //int* idx_sorted_by_width_data_cpu = (int*)malloc(_idx_sorted_by_width->numel() * sizeof(int));
    std::unique_ptr<int[]> idx_sorted_by_width_data_cpu(new int[batch]);

    // sort sequence by width (descending) and find the largest width in the batch
    for (int i = 0; i < batch; i++) {
      width_data[i] = offset[i + 1] - offset[i];
      idx_sorted_by_width_data_cpu[i] = i;
    }
    std::sort(idx_sorted_by_width_data_cpu.get(), idx_sorted_by_width_data_cpu.get() + batch,
              [&_width](int a, int b) {
                return _width.data<int>()[a] > _width.data<int>()[b];
              });
    int max_width = width_data[idx_sorted_by_width_data_cpu[0]];

    // start of reorganizing the input
    std::vector<size_t> new_offset;
    new_offset.resize(max_width + 1);

    new_offset[0] = 0;
    int j = batch - 1;
    int last_width = 0;
    int sub_row = 0;
    int sub_col = 0;

    for (int i = 1; i <= max_width;) {
      for (int k = j; k >= 0; --k) {
        if (width_data[idx_sorted_by_width_data_cpu[k]] > last_width) {
          sub_row = width_data[idx_sorted_by_width_data_cpu[k]] - last_width;
          sub_col = k + 1;

          for (int s = 0; s < sub_row; s++) {
            new_offset[i] = new_offset[i - 1] + sub_col;
            i++;
          }

          // move on
          last_width = width_data[idx_sorted_by_width_data_cpu[k]];
          j = k - 1;
          break;
        }
      }
    }

    // copying to the reorganized buffer
    if (_input->dims().size() == 1) {
      //_layout_input.reshape_batch_sequence({dim0}, new_offset);
    } else {
      //_layout_input.reshape_batch_sequence({dim0, dim1}, new_offset);

      LoD new_lod;
      new_lod.push_back(new_offset);
      _layout_input->set_lod(new_lod);
      _layout_input->Resize({dim0, dim1});
    }

    xpu_memcpy(_idx_sorted_by_width->mutable_data<int>(TARGET(kXPU)),
        idx_sorted_by_width_data_cpu.get(),
        _idx_sorted_by_width->numel() * sizeof(int),
        XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    //auto& dev_ctx = ctx.template device_context<DeviceContext>();
    //memory::Copy(
            //boost::get<platform::XPUPlace>(dev_ctx.GetPlace()),
            //(void*)_idx_sorted_by_width->data<int>(),
            //platform::CPUPlace(),
            //(void*)idx_sorted_by_width_data_cpu,
            //_idx_sorted_by_width->numel() * sizeof(int));

    //std::free(idx_sorted_by_width_data_cpu);
  }

void SearchGrnnCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

    auto* bottom = param.x;
    auto* wi = param.wi;
    auto* wh = param.wh;
    auto* top = param.out;
    auto* _buffer = param.tmp_buffer;
    auto* _idx_sorted_by_width = param.idx_sorted_by_width;
    auto* _layout_input = param.layout_input;
    int _cap_h = param.num_hidden;
    int _cap_e = param.num_input;
    int _cap_l = bottom->dims()[0];

    auto wi_max = param.wi_max;
    auto wh_max = param.wh_max;

    bool float_to_fix = param.float_to_fix;

    CHECK(float_to_fix) << "W should be fixed point";

    int dim = 1;
    if (bottom->dims().size() > 1) {
      dim = bottom->dims()[1];
    }

    const auto& offset = bottom->lod()[0];
    LoD top_lod;
    top_lod.push_back(offset);
    top->set_lod(top_lod);
    std::vector<int64_t> top_dims_vec{_cap_l, _cap_h};
    top->Resize(top_dims_vec);
    auto* top_hidden = top->mutable_data<float>(TARGET(kXPU));
    const auto* dense_e2h = wi->data<int16_t>();
    const auto* dense_h2h = wh->data<int16_t>();

    //auto& dev_ctx = ctx.template device_context<DeviceContext>();

    // Prepare _idx_sorted_by_width
    xpu_prepare_layout(param, bottom);
    int batch = bottom->lod()[0].size() - 1;
    int max_width = _layout_input->lod()[0].size() - 1;
    const auto& new_offset = _layout_input->lod()[0];
    auto* new_emb = _layout_input->mutable_data<float>(TARGET(kXPU));

    // Prepare offset and new_offset
    //int* offset_xpu = nullptr;
    //int* new_offset_xpu = nullptr;
    //float* maxs_xpu = nullptr;
	//offset_xpu = (int*) xpu::alloc_workspace(dev_ctx.x_context(), offset.size() * sizeof(int));
    //new_offset_xpu = (int*) xpu::alloc_workspace(dev_ctx.x_context(), new_offset.size() * sizeof(int));
	//maxs_xpu = (float*) xpu::alloc_workspace(dev_ctx.x_context(), 16 * sizeof(float));
    //PADDLE_ENFORCE(offset_xpu != nullptr, "Fail to alloc L3");
    //PADDLE_ENFORCE(new_offset_xpu != nullptr, "Fail to alloc L3");
    //PADDLE_ENFORCE(maxs_xpu != nullptr, "Fail to alloc L3");
    int* offset_xpu = (int*)offset_xpu_guard_.get();
    int* new_offset_xpu = (int*)new_offset_xpu_guard_.get();
    float* maxs_xpu = (float*)maxs_xpu_guard_.get();

    std::unique_ptr<int[]> offset_cpu(new int[offset.size()]);
    std::unique_ptr<int[]> new_offset_cpu(new int[new_offset.size()]);
    //int* offset_cpu = (int*)malloc(offset.size() * sizeof(int));
    //int* new_offset_cpu = (int*)malloc(new_offset.size() * sizeof(int));
    for (size_t i = 0; i < offset.size(); ++i) {
        offset_cpu[i] = offset[i];
    }
    for (size_t i = 0; i < new_offset.size(); ++i) {
        new_offset_cpu[i] = new_offset[i];
    }
    xpu_memcpy(offset_xpu,
        offset_cpu.get(),
        offset.size() * sizeof(int),
        XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    xpu_memcpy(new_offset_xpu,
        new_offset_cpu.get(),
        new_offset.size() * sizeof(int),
        XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    //memory::Copy(
            //boost::get<platform::XPUPlace>(dev_ctx.GetPlace()),
            //(void*)offset_xpu,
            //platform::CPUPlace(), (void*)offset_cpu,
            //offset.size() * sizeof(int));
    //memory::Copy(
            //boost::get<platform::XPUPlace>(dev_ctx.GetPlace()),
            //(void*)new_offset_xpu,
            //platform::CPUPlace(), (void*)new_offset_cpu,
            //new_offset.size() * sizeof(int));

    // Call xpu seq2batch
    int ret = xdnn::search_seq2batch(ctx.GetRawContext(),
            batch, max_width, dim,
            _idx_sorted_by_width->data<int>(),
            offset_xpu, new_offset_xpu,
            bottom->data<float>(),
            _layout_input->mutable_data<float>());
    //PADDLE_ENFORCE(ret == xpu::Error_t::SUCCESS, "XPU kernel error!");
    (void)ret;

    // this buffer is used for book keeping info which will be used in bp
    // buffer also needed in bp, so make it larger
    _buffer->Resize({20, _cap_l, _cap_h});
    auto* buffer_data = _buffer->mutable_data<float>(TARGET(kXPU));

    // the internal hidden
    auto* hidden = buffer_data + 19 * _cap_l * _cap_h;

    // do-findmax
    float maxs_cpu[16] = {0.0f, 0.0f, 0.0f, 0.0f, wi_max[0], 0.0f, 0.0f, 0.0f,
            wi_max[1], 0.0f, 0.0f, 0.0f, wi_max[2], 0.0f, 0.0f, 0.0f};
    //memory::Copy(boost::get<platform::XPUPlace>(dev_ctx.GetPlace()),
            //(void*)maxs_xpu, platform::CPUPlace(), (void*)maxs_cpu,
            //16 * sizeof(float));
    xpu_memcpy(maxs_xpu,
        maxs_cpu,
        16 * sizeof(float),
        XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    ret = xdnn::findmax<float>(ctx.GetRawContext(), new_emb, _cap_l * _cap_e, maxs_xpu);
    //PADDLE_ENFORCE(ret == xpu::Error_t::SUCCESS, "XPU kernel error!");

    // precompute embedding to hidden
    for (int i = 0; i < 3; ++i) {
        const int16_t* data_b = dense_e2h + i * _cap_e * _cap_h; // e2h, e2hr, e2hz
        float* data_c = buffer_data + i * _cap_l * _cap_h; // w_x_e, wr_x_e, wz_x_e
        int ret = xdnn::gemm_int16_maxptr<float, int16_t, float>(ctx.GetRawContext(),
                false, true,                        // trans_a, trans_b
                _cap_l, _cap_h, _cap_e,             // m, n, k
                1.0f, new_emb, _cap_e,              // alpha, data_a, lda
                data_b, _cap_e, 0.0f,               // data_b, ldb, beta
                data_c, _cap_h,                     // data_c, ldc
                nullptr, xdnn::Activation_t::LINEAR, // bias, act
                maxs_xpu, maxs_xpu + 4 * (i + 1));  // max_a, max_b
        //PADDLE_ENFORCE(ret == xpu::Error_t::SUCCESS, "XPU kernel error!");
        (void)ret;
    }

    // Call xpu search_grnn
    ret = xdnn::search_grnn<float, int16_t>(ctx.GetRawContext(),
            _cap_l, _cap_h, _cap_e,
            max_width, new_offset_xpu,
            buffer_data, dense_h2h, hidden,
            wh_max[0], wh_max[1], wh_max[2]);
    //PADDLE_ENFORCE(ret == xpu::Error_t::SUCCESS, "XPU kernel error!");

    // copy back to top
    ret = xdnn::search_batch2seq(ctx.GetRawContext(),
            batch, max_width, _cap_h,
            _idx_sorted_by_width->data<int>(),
            offset_xpu, new_offset_xpu,
            hidden, top_hidden);
    //PADDLE_ENFORCE(ret == xpu::Error_t::SUCCESS, "XPU kernel error!");

    //std::free(offset_cpu);
    //std::free(new_offset_cpu);

}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(search_grnn,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SearchGrnnCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("tmp_buffer", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("idx_sorted_by_width",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("layout_input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
