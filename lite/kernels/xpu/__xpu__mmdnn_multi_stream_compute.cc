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
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include <algorithm>
#include "assert.h"
#include "xpu/runtime_ex.h"
namespace api = baidu::xpu::api;

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

class L3Manager {
public:
    bool have_l3_workspace;
    char* l3_workspace;
    int l3_size;
    int max_used_l3_size;

    L3Manager(char* _l3_workspace, int _l3_size) {
        have_l3_workspace = (_l3_workspace != nullptr);
        l3_workspace = _l3_workspace;
        l3_size = _l3_size;
        max_used_l3_size = 0;
    }

    ~L3Manager() {}

    float* malloc(int workspace_need_size) {
        float* result = nullptr;
        if (have_l3_workspace && (l3_size >= workspace_need_size)) {
            result = (float*)l3_workspace;
            l3_workspace += workspace_need_size;
            l3_size -= workspace_need_size;
            max_used_l3_size += workspace_need_size;
        }
        return result;
    }
};

class L3BottomTopManager { // can not used in multi stream at the same time
public:
    bool have_l3_workspace;
    // init
    char* l3_workspace_bottom_base;
    char* l3_workspace_top_base;
    // variable
    int bottom_used_l3_size;
    int top_used_l3_size;
    int l3_size_rest;

    L3BottomTopManager(char* _l3_workspace, int _l3_size) {
        have_l3_workspace = (_l3_workspace != nullptr);
        l3_workspace_bottom_base = _l3_workspace;
        l3_workspace_top_base = _l3_workspace + _l3_size;
        bottom_used_l3_size = 0;
        top_used_l3_size = 0;
        l3_size_rest = _l3_size;
    }

    ~L3BottomTopManager() {}

    float* malloc_bottom(int workspace_need_size) {
        float* result = nullptr;
        if (have_l3_workspace && (l3_size_rest >= workspace_need_size)) {
            result = (float*)(l3_workspace_bottom_base + bottom_used_l3_size);
            bottom_used_l3_size += workspace_need_size;
            l3_size_rest -= workspace_need_size;
        }
        return result;
    }

    void free_bottom() {
        l3_size_rest += bottom_used_l3_size;
        bottom_used_l3_size = 0;
    }

    float* malloc_top(int workspace_need_size) {
        float* result = nullptr;
        if (have_l3_workspace && (l3_size_rest >= workspace_need_size)) {
            top_used_l3_size += workspace_need_size;
            l3_size_rest -= workspace_need_size;
            result = (float*)(l3_workspace_top_base - top_used_l3_size);
        }
        return result;
    }

    void free_top() {
        l3_size_rest += top_used_l3_size;
        top_used_l3_size = 0;
    }

    char* unused_workspace() {
        return l3_workspace_bottom_base + bottom_used_l3_size;
    }

    int unused_size() {
        return l3_size_rest;
    }
};

void grnn_layout(std::vector<int> width, std::vector<int>& new_offset, std::vector<int>& idx_sorted) {
    new_offset.clear();
    idx_sorted.clear();

    int batch = width.size();
    idx_sorted.resize(batch);
    for (int i = 0; i < batch; i++) {
        idx_sorted[i] = i;
    }
    std::sort(idx_sorted.data(), idx_sorted.data() + batch,
        [&width](int a, int b) {
            return width[a] > width[b];
        });
    int max_width = width[idx_sorted[0]];
    new_offset.resize(max_width + 1);
    new_offset[0] = 0;
    int j = batch - 1;
    int last_width = 0;
    int sub_row = 0;
    int sub_col = 0;

    for (int i = 1; i <= max_width;) {
        for (int k = j; k >= 0; --k) {
            if (width[idx_sorted[k]] > last_width) {
                sub_row = width[idx_sorted[k]] - last_width;
                sub_col = k + 1;
                for (int s = 0; s < sub_row; s++) {
                    new_offset[i] = new_offset[i - 1] + sub_col;
                    i++;
                }
                // move on
                last_width = width[idx_sorted[k]];
                j = k - 1;
                break;
            }
        }
    }
}

class IdInfo{
    char* xpu_buffer_hbm{nullptr};
    char* xpu_buffer_l3{nullptr};
    char* xpu_buffer{nullptr};
    char* cpu_buffer{nullptr};
    int buffer_size{0};
public:
    int64_t* id0_64; // UB_batch * UB_seqlen * sizeof(int64_t)
    int64_t* id1_64; // UB_batch * UB_seqlen * sizeof(int64_t)
    int64_t* lod_64; // (UB_batch + 1) * sizeof(int64_t)
    int* lod_32; // (UB_batch + 1) * sizeof(int)
    int* new_offset_32; // (UB_seqlen + 1) * sizeof(int)
    int* idx_sorted_32; // (UB_batch + 1) * sizeof(int)

    std::vector<int64_t> lod_64_cpu;
    std::vector<int> lod_32_cpu;
    std::vector<int> new_offset;
    std::vector<int> idx_sorted;

    int batch;
    int seqlen_max;
    int seqlen_sum;
    int seqlen_square_sum;
    std::vector<int> seqlen_list;

    void init(int UB_batch, int UB_seqlen) {
        buffer_size = (UB_batch + 1) * sizeof(int64_t) + \
                        (UB_batch + 1) * sizeof(int) * 2 + \
                        (UB_seqlen + 1) * sizeof(int);

        xpu_malloc((void**)(&xpu_buffer_hbm), buffer_size);
        xpu_buffer = xpu_buffer_hbm;
        cpu_buffer = (char*)malloc(buffer_size);
    }

    void get_xpu_buffer_l3(char* &l3_workspace, int &l3_size) {
        if ((l3_workspace != nullptr) && (l3_size >= buffer_size)) {
            xpu_buffer_l3 = l3_workspace;
            xpu_buffer = xpu_buffer_l3;
            l3_workspace += buffer_size;
            l3_size -= buffer_size;
        }
    }

    void update(lite::Tensor* id0, lite::Tensor* id1) {
        id0_64 = id0->mutable_data<int64_t>();
        id1_64 = id1->mutable_data<int64_t>();

        lod_64_cpu.clear();
        lod_32_cpu.clear();
        for (auto e : id0->lod()[0]) {
            lod_64_cpu.push_back(e);
            lod_32_cpu.push_back(e);
        }

        batch = lod_32_cpu.size() - 1;
        seqlen_max = 0;
        seqlen_sum = 0;
        seqlen_square_sum = 0;
        seqlen_list.resize(batch);
        for (int i = 0; i < batch; i++) {
            int seqlen = lod_32_cpu[i + 1] - lod_32_cpu[i];
            seqlen_max = std::max(seqlen_max, seqlen);
            seqlen_sum = seqlen_sum + seqlen;
            seqlen_square_sum = seqlen_square_sum + seqlen * seqlen;
            seqlen_list[i] = seqlen;
        }

        grnn_layout(seqlen_list, new_offset, idx_sorted);

        int offset = 0;

        lod_64 = (int64_t*)(xpu_buffer + offset);
        memcpy(cpu_buffer + offset, lod_64_cpu.data(), lod_64_cpu.size() * sizeof(int64_t));
        offset += lod_64_cpu.size() * sizeof(int64_t);

        lod_32 = (int*)(xpu_buffer + offset);
        memcpy(cpu_buffer + offset, lod_32_cpu.data(), lod_32_cpu.size() * sizeof(int));
        offset += lod_32_cpu.size() * sizeof(int);

        new_offset_32 = (int*)(xpu_buffer + offset);
        memcpy(cpu_buffer + offset, new_offset.data(), new_offset.size() * sizeof(int));
        offset += new_offset.size() * sizeof(int);

        idx_sorted_32 = (int*)(xpu_buffer + offset);
        memcpy(cpu_buffer + offset, idx_sorted.data(), idx_sorted.size() * sizeof(int));
        offset += idx_sorted.size() * sizeof(int);

        xpu_memcpy((void*)xpu_buffer,
            (void*)cpu_buffer,
            offset,
            XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    }
};
/***** API *****/
class BidEmbEw{
    float* table;
    int table_len;
    int emb_dim;
    // hbm_buffer
    float* out_fw_hbm;
    float* out_rv_hbm;
public:
    float* out_fw; // [cap_l, emb_dim]
    float* out_rv; // [cap_l, emb_dim]

    void init(float* _table, int _table_len, int _emb_dim,
            int max_cap_l) {
        table = _table;
        table_len = _table_len;
        emb_dim = _emb_dim;
        // hbm_buffer
        xpu_malloc((void**)(&out_fw_hbm), max_cap_l * emb_dim * sizeof(float));
        xpu_malloc((void**)(&out_rv_hbm), max_cap_l * emb_dim * sizeof(float));
    }

    int out_fw_size(const IdInfo& sentense) {
        return sentense.seqlen_sum * emb_dim * sizeof(float);
    }

    int out_rv_size(const IdInfo& sentense) {
        return sentense.seqlen_sum * emb_dim * sizeof(float);
    }

    int total_result_size(const IdInfo& sentense) {
        return out_fw_size(sentense) + out_rv_size(sentense);
    }

    void infer(api::Context* ctx, const IdInfo& sentense,
            float* out_fw_l3 = nullptr, float* out_rv_l3 = nullptr) {
        out_fw = (out_fw_l3 == nullptr) ? out_fw_hbm : out_fw_l3;
        out_rv = (out_rv_l3 == nullptr) ? out_rv_hbm : out_rv_l3;
        api::search_bid_emb_ew(ctx,
            sentense.batch, sentense.lod_64, sentense.id0_64, sentense.id1_64,
            table, table_len, emb_dim, out_fw, out_rv, table_len - 2, 1);
    }
};

class Seq2Batch {
    int cap_e;
    // hbm_buffer
    float* out_hbm;
public:
    float* out; // [cap_l, cap_e]

    void init(int _cap_e, int max_cap_l) {
        cap_e = _cap_e;
        // hbm_buffer
        xpu_malloc((void**)(&out_hbm), max_cap_l * cap_e * sizeof(float));
    }

    int out_size(const IdInfo& sentense) {
        return sentense.seqlen_sum * cap_e * sizeof(float);
    }

    int total_result_size(const IdInfo& sentense) {
        return out_size(sentense);
    }

    void infer(api::Context* ctx, const IdInfo& sentense,
            const float* in, float* out_l3 = nullptr) {
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        api::search_seq2batch(ctx,
                sentense.batch, sentense.seqlen_max, cap_e,
                sentense.idx_sorted_32, sentense.lod_32, sentense.new_offset_32,
                in, out);
    }
};

class FindMax {
    // hbm_buffer
    float* out_hbm;
public:
    float* out; // [4]

    void init() {
        // hbm_buffer
        xpu_malloc((void**)(&out_hbm), 4 * sizeof(float));
    }

    int out_size() {
        return 4 * sizeof(float);
    }

    int total_result_size() {
        return out_size();
    }

    void infer(api::Context* ctx, const float* in, int size,
            float* out_l3 = nullptr) {
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        api::findmax<float>(ctx, in, size, out);
    }
};

class FC {
    int max_m;
    int n;
    int k;
    int16_t* weight;
    float* weight_max;
    float* bias;
    api::Activation_t::act_enum act_type;
    // hbm_buffer
    float* out_hbm;
    float* out_max_hbm;
public:
    float* out; // [m, n]
    float* out_max; // [4]

    void init(int _max_m, int _n, int _k,
            int16_t* _weight, float _weight_max,
            float* _bias = nullptr,
            api::Activation_t::act_enum _act_type = api::Activation_t::LINEAR) {
        max_m = _max_m;
        n = _n;
        k = _k;
        weight = _weight;
        xpu_malloc((void**)(&weight_max), 4 * sizeof(float));
        float weight_max_cpu[4] = {_weight_max, 0.0f, 0.0f, 0.0f};
        xpu_memcpy((void*)(weight_max),
            (void*)weight_max_cpu,
            4 * sizeof(float),
            XPUMemcpyKind::XPU_HOST_TO_DEVICE);
        bias = _bias;
        act_type = _act_type;
        // hbm_buffer
        xpu_malloc((void**)(&out_hbm), max_m * n * sizeof(float));
        xpu_malloc((void**)(&out_max_hbm), 4 * sizeof(float));
    }

    int out_size(int m) {
        return m * n * sizeof(float);
    }

    int out_max_size() {
        return 4 * sizeof(float);
    }

    int total_result_size(int m) {
        return out_size(m) + out_max_size();
    }

    void infer(api::Context* ctx,
            const float* in, const float* in_max, int m,
            float* out_l3 = nullptr, float* out_max_l3 = nullptr) {
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        out_max = (out_max_l3 == nullptr) ? out_max_hbm : out_max_l3;
        api::fc_int16(ctx, false, true,
            m, n, k,
            1.0f,
            in, in_max,
            weight, weight_max,
            0.0f,
            out, out_max,
            bias, act_type);
    }
};

class SearchBiGrnn {
    int cap_e;
    int cap_h;
    // fw_weight
    int16_t* fw_dense_h2h;
    float fw_dense_h2h_max[3];
    // rv_weight
    int16_t* rv_dense_h2h;
    float rv_dense_h2h_max[3];
    // hbm_buffer
    float* fw_out_hbm;
    float* rv_out_hbm;
public:
    float* fw_out; // [cap_l, cap_h]
    float* rv_out; // [cap_l, cap_h]

    void init(
            int16_t* _fw_dense_h2h, std::vector<float>& _fw_dense_h2h_max,
            int16_t* _rv_dense_h2h, std::vector<float>& _rv_dense_h2h_max,
            int _cap_e, int _cap_h, int max_cap_l) {
        cap_e = _cap_e;
        cap_h = _cap_h;
        // fw_weight
        fw_dense_h2h = _fw_dense_h2h;
        for (int i = 0; i < 3; i++) {
            fw_dense_h2h_max[i] = _fw_dense_h2h_max[i];
        }
        // rv_weight
        rv_dense_h2h = _rv_dense_h2h;
        for (int i = 0; i < 3; i++) {
            rv_dense_h2h_max[i] = _rv_dense_h2h_max[i];
        }
        // hbm_buffer
        xpu_malloc((void**)(&fw_out_hbm), max_cap_l * cap_h * sizeof(float));
        xpu_malloc((void**)(&rv_out_hbm), max_cap_l * cap_h * sizeof(float));
    }

    int fw_out_size(const IdInfo& sentense) {
        return sentense.seqlen_sum * cap_h * sizeof(float);
    }

    int rv_out_size(const IdInfo& sentense) {
        return sentense.seqlen_sum * cap_h * sizeof(float);
    }

    int total_result_size(const IdInfo& sentense) {
        return fw_out_size(sentense) + rv_out_size(sentense);
    }

    void infer(api::Context* ctx, const IdInfo& sentense,
            const float* fw_in, const float* rv_in,
            float* fw_out_l3 = nullptr, float* rv_out_l3 = nullptr) {
        fw_out = (fw_out_l3 == nullptr) ? fw_out_hbm : fw_out_l3;
        rv_out = (rv_out_l3 == nullptr) ? rv_out_hbm : rv_out_l3;
        api::search_bi_grnn<float, int16_t>(ctx,
                sentense.seqlen_sum, cap_h, sentense.seqlen_max, sentense.new_offset_32,
                fw_in, fw_dense_h2h, fw_out,
                fw_dense_h2h_max[0], fw_dense_h2h_max[1], fw_dense_h2h_max[2],
                rv_in, rv_dense_h2h, rv_out,
                rv_dense_h2h_max[0], rv_dense_h2h_max[1], rv_dense_h2h_max[2]);
    }
};

class Batch2Seq {
    int cap_h;
    // hbm_buffer
    float* out_hbm;
public:
    float* out; // [cap_l, cap_h]

    void init(int _cap_h, int max_cap_l) {
        cap_h = _cap_h;
        // hbm_buffer
        xpu_malloc((void**)(&out_hbm), max_cap_l * cap_h * sizeof(float));
    }

    int out_size(const IdInfo& sentense) {
        return sentense.seqlen_sum * cap_h * sizeof(float);
    }

    int total_result_size(const IdInfo& sentense) {
        return out_size(sentense);
    }

    void infer(api::Context* ctx, const IdInfo& sentense,
            const float* in, float* out_l3 = nullptr) {
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        api::search_batch2seq(ctx,
                sentense.batch, sentense.seqlen_max, cap_h,
                sentense.idx_sorted_32, sentense.lod_32, sentense.new_offset_32,
                in, out);
    }
};

class SeqLastPooling {
    int dim;
    // hbm_buffer
    float* out_hbm;
public:
    float* out; // [batch, dim]

    void init(int _dim, int UB_batch) {
        dim = _dim;
        // hbm_buffer
        xpu_malloc((void**)(&out_hbm), UB_batch * dim * sizeof(float));
    }

    int out_size(const IdInfo& sentense) {
        return sentense.batch * dim * sizeof(float);
    }

    int total_result_size(const IdInfo& sentense) {
        return out_size(sentense);
    }

    void infer(api::Context* ctx, const IdInfo& sentense,
            const float* in, float* out_l3 = nullptr) {
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        api::sequence_pooling_forward(ctx,
            api::Pooling_t::LAST,
            sentense.batch, sentense.lod_32, dim,
            in, nullptr, out);
    }
};

class SeqReverse {
    int dim;
    // hbm_buffer
    float* out_hbm;
public:
    float* out; // [cap_l, dim]

    void init(int _dim, int max_cap_l) {
        dim = _dim;
        // hbm_buffer
        xpu_malloc((void**)(&out_hbm), max_cap_l * dim * sizeof(float));
    }

    int out_size(const IdInfo& sentense) {
        return sentense.seqlen_sum * dim * sizeof(float);
    }

    int total_result_size(const IdInfo& sentense) {
        return out_size(sentense);
    }


    void infer(api::Context* ctx, const IdInfo& sentense,
            const float* in, float* out_l3 = nullptr) {
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        api::sequence_reverse(ctx,
            sentense.batch, sentense.lod_32, dim,
            in, out);
    }
};


class Concat2In {
    int concat_widths[2];
    // hbm_buffer
    float* out_hbm;
public:
    float* out; // [h, w1 + w2]

    void init(int max_h, int w1, int w2) {
        concat_widths[0] = w1;
        concat_widths[1] = w2;
        // hbm_buffer
        xpu_malloc((void**)(&out_hbm),
            max_h * (w1 + w2) * sizeof(float));
    }

    int out_size(int h) {
        return h * (concat_widths[0] + concat_widths[1]) * sizeof(float);
    }

    int total_result_size(int h) {
        return out_size(h);
    }

    void infer(api::Context* ctx, int h,
            const float* in1, const float* in2,
            float* out_l3 = nullptr) {
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        const float* concat_ptrs[] = {in1, in2};
        api::concat<float>(ctx, h, concat_widths, 2, concat_ptrs, out);
    }
};

class Concat3In {
    int concat_widths[3];
    // hbm_buffer
    float* out_hbm;
public:
    float* out; // [h, w1 + w2 + w3]

    void init(int max_h, int w1, int w2, int w3) {
        concat_widths[0] = w1;
        concat_widths[1] = w2;
        concat_widths[2] = w3;
        // hbm_buffer
        xpu_malloc((void**)(&out_hbm),
            max_h * (w1 + w2 + w3) * sizeof(float));
    }

    int out_size(int h) {
        return h * (concat_widths[0]
            + concat_widths[1]
            + concat_widths[2]) * sizeof(float);
    }

    int total_result_size(int h) {
        return out_size(h);
    }

    void infer(api::Context* ctx, int h,
            const float* in1, const float* in2, const float* in3,
            float* out_l3 = nullptr) {
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        const float* concat_ptrs[] = {in1, in2, in3};
        api::concat<float>(ctx, h, concat_widths, 3, concat_ptrs, out);
    }
};

class Concat7In {
    int concat_widths[7];
    // hbm_buffer
    float* out_hbm;
public:
    float* out; // [h, w1 + w2 + w3 + w4 + w5 + w6 + w7]

    void init(int max_h, int w1, int w2, int w3, int w4, int w5, int w6, int w7) {
        concat_widths[0] = w1;
        concat_widths[1] = w2;
        concat_widths[2] = w3;
        concat_widths[3] = w4;
        concat_widths[4] = w5;
        concat_widths[5] = w6;
        concat_widths[6] = w7;
        // hbm_buffer
        xpu_malloc((void**)(&out_hbm),
            max_h * (w1 + w2 + w3 + w4 + w5 + w6 + w7) * sizeof(float));
    }

    int out_size(int h) {
        return h * (concat_widths[0]
            + concat_widths[1]
            + concat_widths[2]
            + concat_widths[3]
            + concat_widths[4]
            + concat_widths[5]
            + concat_widths[6]) * sizeof(float);
    }

    int total_result_size(int h) {
        return out_size(h);
    }

    void infer(api::Context* ctx, int h,
            const float* in1,
            const float* in2,
            const float* in3,
            const float* in4,
            const float* in5,
            const float* in6,
            const float* in7,
            float* out_l3 = nullptr) {
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        const float* concat_ptrs[] = {in1, in2, in3, in4, in5, in6, in7};
        api::concat<float>(ctx, h, concat_widths, 7, concat_ptrs, out);
    }
};

class EmbEw {
    float* table;
    int table_len;
    int emb_dim;
    // hbm_buffer
    float* out_hbm;
public:
    float* out; // [cap_l, emb_dim]

    void init(lite::Tensor* table_tensor, int UB_batch, int UB_seqlen) {
        table = table_tensor->mutable_data<float>();
        table_len = table_tensor->dims()[0];
        emb_dim = table_tensor->dims()[1];
        // hbm_buffer
        int max_cap_len = UB_batch * UB_seqlen;
        xpu_malloc((void**)(&out_hbm), max_cap_len * emb_dim * sizeof(float));
    }

    int out_size(const IdInfo& sentense) {
        return sentense.seqlen_sum * emb_dim * sizeof(float);
    }

    int total_result_size(const IdInfo& sentense) {
        return out_size(sentense);
    }

    void infer(api::Context* ctx, const IdInfo& sentense, float* out_l3 = nullptr) {
        int cap_l = sentense.seqlen_sum;
        const float* emb_tables[] = {table, table};
        const int64_t* emb_indices[] = {sentense.id0_64, sentense.id1_64};
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        api::embedding_with_ewadd<float, int64_t, false, false>(ctx,
            emb_dim, cap_l, 2, table_len - 2,
            emb_tables, emb_indices, nullptr, nullptr, out);
    }
};

class MatchMatrixTensor {
    int dim_t;
    int dim_in;
    // hbm_buffer
    float* out_hbm;
public:
    float* out; // [sum(left_len * right_len) * dim_t]

    void init(int _dim_t, int _dim_in, int UB_batch, int UB_seqlen) {
        dim_t = _dim_t;
        dim_in = _dim_in;
        // hbm_buffer
        xpu_malloc((void**)(&out_hbm),
            UB_batch * UB_seqlen * UB_seqlen * dim_t * sizeof(float));
    }

    int out_size(const IdInfo& left_sentense, const IdInfo& right_sentense) {
        assert(left_sentense.batch == right_sentense.batch);
        int batch = left_sentense.batch;
        int x_mul_y_sum = 0;
        for (int i = 0; i < batch; i++) {
            x_mul_y_sum += left_sentense.seqlen_list[i] * right_sentense.seqlen_list[i];
        }
        return x_mul_y_sum * dim_t * sizeof(float);
    }

    int total_result_size(const IdInfo& left_sentense, const IdInfo& right_sentense) {
        return out_size(left_sentense, right_sentense);
    }

    void infer(api::Context* ctx,
            const IdInfo& left_sentense, const IdInfo& right_sentense,
            const float* x_dot_w_in, const float* x_dot_w_in_max, const float* y,
            float* out_l3 = nullptr) {
        assert(left_sentense.batch == right_sentense.batch);
        int batch = left_sentense.batch;
        int* left_lod_32 = left_sentense.lod_32;
        int* right_lod_32 = right_sentense.lod_32;
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        int max_width = std::max(left_sentense.seqlen_max, right_sentense.seqlen_max);
        api::match_matrix_tensor(ctx,
            batch, x_dot_w_in, y, left_lod_32, right_lod_32,
            dim_t, dim_in, out, x_dot_w_in_max, api::Activation_t::RELU, max_width);
    }
};

class SearchVarConv {
    int dim_t;
    int out_channel;
    int16_t* conv_weight;
    float conv_weight_max;
    // hbm_buffer
    float* out_hbm;
public:
    float* out; // [sum(left_len * right_len) * out_channel]

    void init(int16_t* _conv_weight, float _conv_weight_max,
            int _dim_t, int _out_channel, int UB_batch, int UB_seqlen) {
        dim_t = _dim_t;
        out_channel = _out_channel;
        conv_weight = _conv_weight;
        conv_weight_max = _conv_weight_max;
        // hbm_buffer
        xpu_malloc((void**)(&out_hbm),
            UB_batch * UB_seqlen * UB_seqlen * out_channel * sizeof(float));
    }

    int out_size(const IdInfo& left_sentense, const IdInfo& right_sentense) {
        assert(left_sentense.batch == right_sentense.batch);
        int batch = left_sentense.batch;
        int x_mul_y_sum = 0;
        for (int i = 0; i < batch; i++) {
            x_mul_y_sum += left_sentense.seqlen_list[i] * right_sentense.seqlen_list[i];
        }
        return x_mul_y_sum * out_channel * sizeof(float);
    }

    int total_result_size(const IdInfo& left_sentense, const IdInfo& right_sentense) {
        return out_size(left_sentense, right_sentense);
    }

    void infer(api::Context* ctx,
            const IdInfo& left_sentense, const IdInfo& right_sentense,
            const float* in, // xwy_out
            float* out_l3 = nullptr) {
        assert(left_sentense.batch == right_sentense.batch);
        int batch = left_sentense.batch;
        int* left_lod_32 = left_sentense.lod_32;
        int* right_lod_32 = right_sentense.lod_32;
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        api::search_varconv<float, int16_t>(ctx,
            batch, dim_t, out_channel, 5, 5, 1, 1,
            in, conv_weight,
            right_lod_32, left_lod_32,
            out, conv_weight_max,
            api::Activation_t::RELU); // x=right, y=left
    }
};

class SeqConcat {
    int dim_t;
    int out_channel;
    //
    int* match_lod_32;
    int* varconv_lod_32;
    //
    float* out_hbm;
public:
    float* out; // [sum(left_len * right_len) * (dim_t + out_channel)]

    void init(int _dim_t, int _out_channel, int UB_batch, int UB_seqlen) {
        dim_t = _dim_t;
        out_channel = _out_channel;
        //
        xpu_malloc((void**)(&match_lod_32), (UB_batch + 1) * sizeof(int));
        xpu_malloc((void**)(&varconv_lod_32), (UB_batch + 1) * sizeof(int));
        //
        xpu_malloc((void**)(&out_hbm),
            UB_batch * UB_seqlen * UB_seqlen * (dim_t + out_channel) * sizeof(float));
    }

    int out_size(const IdInfo& left_sentense, const IdInfo& right_sentense) {
        assert(left_sentense.batch == right_sentense.batch);
        int batch = left_sentense.batch;
        int x_mul_y_sum = 0;
        for (int i = 0; i < batch; i++) {
            x_mul_y_sum += left_sentense.seqlen_list[i] * right_sentense.seqlen_list[i];
        }
        return x_mul_y_sum * (dim_t + out_channel) * sizeof(float);
    }

    int total_result_size(const IdInfo& left_sentense, const IdInfo& right_sentense) {
        return out_size(left_sentense, right_sentense);
    }

    void infer(api::Context* ctx,
            const IdInfo& left_sentense, const IdInfo& right_sentense,
            const float* xwy_out, const float* varconv_out,
            float* out_l3 = nullptr) {
        assert(left_sentense.batch == right_sentense.batch);
        int batch = left_sentense.batch;
        std::vector<int> lod_match = {0};
        std::vector<int> lod_varconv = {0};
        for (int i = 0; i < batch; i++) {
            int imgsize = left_sentense.seqlen_list[i] * right_sentense.seqlen_list[i];
            lod_match.push_back(lod_match.back() + imgsize * dim_t);
            lod_varconv.push_back(lod_varconv.back() + imgsize * out_channel);
        }
        xpu_memcpy((void*)match_lod_32,
                (void*)lod_match.data(),
                lod_match.size() * sizeof(int),
                XPUMemcpyKind::XPU_HOST_TO_DEVICE);
        xpu_memcpy((void*)varconv_lod_32,
                (void*)lod_varconv.data(),
                lod_varconv.size() * sizeof(int),
                XPUMemcpyKind::XPU_HOST_TO_DEVICE);
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        api::sequence_concat(ctx,
            xwy_out, match_lod_32,
            varconv_out, varconv_lod_32,
            out, batch);
    }
};

int* get_constant_int32_xpu(std::vector<int> int_cpu) {
    int* buffer_xpu = nullptr;
    xpu_malloc((void**)(&buffer_xpu),
        int_cpu.size() * sizeof(int));
    xpu_memcpy((void*)buffer_xpu,
        (void*)int_cpu.data(),
        int_cpu.size() * sizeof(int),
        XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    return buffer_xpu;
}

class TopkAvgPooling{
    int dim_t;
    int out_channel;
    std::vector<int> topks;
    //
    int* useless_topk_pos; // [4]
    int* topk_offset_32;
    int* topks_xpu;
    //
    float* out_hbm;
public:
    float* out; // [left_cap_l, (dim_t + out_channel), topks.size()]

    void init(int _dim_t, int _out_channel, int UB_batch, int UB_seqlen, std::vector<int> _topks) {
        dim_t = _dim_t;
        out_channel = _out_channel;
        topks = _topks;
        //
        xpu_malloc((void**)(&useless_topk_pos), 4 * sizeof(int));
        xpu_malloc((void**)(&topk_offset_32), (UB_batch + 1) * sizeof(int));
        topks_xpu = get_constant_int32_xpu(topks);
        //
        xpu_malloc((void**)(&out_hbm),
                UB_batch * UB_seqlen * topks.size() * (dim_t + out_channel) * sizeof(float));
    }

    int out_size(const IdInfo& left_sentense, const IdInfo& right_sentense) {
        return left_sentense.seqlen_sum * (dim_t + out_channel) * topks.size() * sizeof(float);
    }

    int total_result_size(const IdInfo& left_sentense, const IdInfo& right_sentense) {
        return out_size(left_sentense, right_sentense);
    }

    void infer(api::Context* ctx,
            const IdInfo& left_sentense, const IdInfo& right_sentense,
            const float* in, float* out_l3 = nullptr) {
        assert(left_sentense.batch == right_sentense.batch);
        int batch = left_sentense.batch;
        //
        std::vector<int> lod_topk = {0};
        for (int i = 0; i < batch; i++) {
            int imgsize = left_sentense.seqlen_list[i] * right_sentense.seqlen_list[i];
            lod_topk.push_back(lod_topk.back() + imgsize * (dim_t + out_channel));
        }
        xpu_memcpy((void*)topk_offset_32,
                (void*)lod_topk.data(),
                lod_topk.size() * sizeof(int),
                XPUMemcpyKind::XPU_HOST_TO_DEVICE);
        // get lods
        int* left_lod_32 = left_sentense.lod_32;
        int* right_lod_32 = right_sentense.lod_32;
        // avg-topk
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        api::sequence_topk_avg_pooling(ctx,
            in, out, useless_topk_pos,
            batch, dim_t + out_channel,
            topk_offset_32, left_lod_32, right_lod_32, topks_xpu, topks.size());
    }
};
/***** OP *****/
class BiGrnnBegin {
public:
    BidEmbEw bid_emb_ew;
    Seq2Batch fw_seq2batch;
    Seq2Batch rv_seq2batch;

    void init(lite::Tensor* table_tensor, int UB_batch, int UB_seqlen) {
        float* _table = table_tensor->mutable_data<float>();
        int _table_len = table_tensor->dims()[0];
        int _emb_dim = table_tensor->dims()[1];
        int _cap_e = _emb_dim;
        int max_cap_l = UB_batch * UB_seqlen;
        bid_emb_ew.init(_table, _table_len, _emb_dim, max_cap_l);
        fw_seq2batch.init(_cap_e, max_cap_l);
        rv_seq2batch.init(_cap_e, max_cap_l);
    }

    int infer(api::Context* ctx, const IdInfo& sentense,
            char* l3_workspace = nullptr, int l3_size = 0,
            float* bid_emb_ew_out_fw_l3 = nullptr,
            float* fw_seq2batch_out_l3 = nullptr,
            float* rv_seq2batch_out_l3 = nullptr) {
        int max_used_l3_size = 0;
        float* bid_emb_ew_out_rv_l3 = nullptr;
        int workspace_need_size = bid_emb_ew.out_rv_size(sentense);
        if ((l3_workspace != nullptr) && (l3_size > workspace_need_size)) {
            bid_emb_ew_out_rv_l3 = (float*)l3_workspace;
            //l3_workspace += workspace_need_size;
            //l3_size -= workspace_need_size;
            max_used_l3_size += workspace_need_size;
        }
        bid_emb_ew.infer(ctx, sentense,
            bid_emb_ew_out_fw_l3, bid_emb_ew_out_rv_l3);
        fw_seq2batch.infer(ctx, sentense, bid_emb_ew.out_fw,
            fw_seq2batch_out_l3);
        rv_seq2batch.infer(ctx, sentense, bid_emb_ew.out_rv,
            rv_seq2batch_out_l3);
        return max_used_l3_size;
    }
};

class BiGrnnCDNN {
    int cap_e;
    int cap_h;
public:
    FindMax input_find_max;
    FC fw_fc_e2h;
    FC rv_fc_e2h;
    SearchBiGrnn bi_grnn;

    void init(lite::Tensor* fw_wh, std::vector<float>& fw_wh_maxs,
            lite::Tensor* fw_wi, std::vector<float>& fw_wi_maxs,
            lite::Tensor* rv_wh, std::vector<float>& rv_wh_maxs,
            lite::Tensor* rv_wi, std::vector<float>& rv_wi_maxs,
            int _cap_e, int _cap_h,
            int UB_batch, int UB_seqlen) {
        cap_e = _cap_e;
        cap_h = _cap_h;
        int max_cap_l = UB_batch * UB_seqlen;
        // fw
        input_find_max.init();
        // weight-e2h
        fw_fc_e2h.init(max_cap_l, 3 * cap_h, cap_e,
            fw_wi->mutable_data<int16_t>(), fw_wi_maxs[0]);
        rv_fc_e2h.init(max_cap_l, 3 * cap_h, cap_e,
            rv_wi->mutable_data<int16_t>(), rv_wi_maxs[0]);
        // weight-h2h
        bi_grnn.init(
            fw_wh->mutable_data<int16_t>(), fw_wh_maxs,
            rv_wh->mutable_data<int16_t>(), rv_wh_maxs,
            cap_e, cap_h, max_cap_l);
    }

    int total_result_size(const IdInfo& sentense) {
        return bi_grnn.total_result_size(sentense);
    }

    int infer(api::Context* ctx, const IdInfo& sentense,
            const float* fw_seq2batch, const float* rv_seq2batch,
            char* l3_workspace = nullptr, int l3_size = 0,
            float* fw_out_l3 = nullptr, float* rv_out_l3 = nullptr) {
        L3Manager l3_manager(l3_workspace, l3_size);
        int cap_l = sentense.seqlen_sum;
        // find max
        int workspace_need_size = input_find_max.total_result_size();
        input_find_max.infer(ctx, fw_seq2batch, cap_l * cap_e,
            l3_manager.malloc(workspace_need_size));
        // fw_fc
        workspace_need_size = (4 + 3 * cap_l * cap_h) * sizeof(float);
        float* fw_fc_e2h_out_max_l3 = l3_manager.malloc(workspace_need_size);
        float* fw_fc_e2h_out_l3 = (fw_fc_e2h_out_max_l3 == nullptr) ? nullptr : (fw_fc_e2h_out_max_l3 + 4);
        fw_fc_e2h.infer(ctx, fw_seq2batch, input_find_max.out, cap_l,
            fw_fc_e2h_out_l3, fw_fc_e2h_out_max_l3);
        // rv_fc
        //workspace_need_size = (4 + 3 * cap_l * cap_h) * sizeof(float);
        float* rv_fc_e2h_out_max_l3 = l3_manager.malloc(workspace_need_size);
        float* rv_fc_e2h_out_l3 = (rv_fc_e2h_out_max_l3 == nullptr) ? nullptr : (rv_fc_e2h_out_max_l3 + 4);
        rv_fc_e2h.infer(ctx, rv_seq2batch, input_find_max.out, cap_l,
            rv_fc_e2h_out_l3, rv_fc_e2h_out_max_l3);
        // search_grnn
        bi_grnn.infer(ctx, sentense, fw_fc_e2h.out, rv_fc_e2h.out,
            fw_out_l3, rv_out_l3);
        return l3_manager.max_used_l3_size;
    }
};

class BiGrnnEnd {
public:
    Batch2Seq rv_batch2seq; // internal [cap_l, cap_h]
    SeqLastPooling rv_seq_last_pooling;
    SeqReverse seq_reverse; // internal [cap_l, cap_h]

    Batch2Seq fw_batch2seq; // internal [cap_l, cap_h] reuse rv_batch2seq.out
    SeqLastPooling fw_seq_last_pooling;

    Concat2In concat_2in;
    Concat3In concat_3in;

    void init(int _cap_e, int _cap_h, int UB_batch, int UB_seqlen) {
        int max_cap_l = UB_batch * UB_seqlen;
        rv_batch2seq.init(_cap_h, max_cap_l);
        seq_reverse.init(_cap_h, max_cap_l);
        rv_seq_last_pooling.init(_cap_h, UB_batch);
        fw_batch2seq.init(_cap_h, max_cap_l);
        fw_seq_last_pooling.init(_cap_h, UB_batch);
        concat_2in.init(max_cap_l, _cap_h, _cap_h);
        concat_3in.init(max_cap_l, _cap_e, _cap_h, _cap_h);
    }

    int infer(api::Context* ctx, const IdInfo& sentense,
            const float* bid_emb_ew_fw_out, const float* grnn_cdnn_fw_out, const float* grnn_cdnn_rv_out,
            char* l3_workspace = nullptr, int l3_size = 0,
            float* fw_pool_out_l3 = nullptr,
            float* rv_pool_out_l3 = nullptr,
            float* concat_2in_out_l3 = nullptr,
            float* concat_3in_out_l3 = nullptr) {
        L3Manager l3_manager(l3_workspace, l3_size);
        // rv_batch2seq
        int workspace_need_size = rv_batch2seq.total_result_size(sentense);
        rv_batch2seq.infer(ctx, sentense, grnn_cdnn_rv_out,
            l3_manager.malloc(workspace_need_size));
        // seq_reverse
        workspace_need_size = seq_reverse.total_result_size(sentense);
        seq_reverse.infer(ctx, sentense, rv_batch2seq.out,
            l3_manager.malloc(workspace_need_size));
        // rv_seq_last_pooling
        rv_seq_last_pooling.infer(ctx, sentense, rv_batch2seq.out,
            rv_pool_out_l3);
        // fw_batch2seq
        //assert(fw_batch2seq.total_result_size == rv_batch2seq.total_result_size)
        fw_batch2seq.infer(ctx, sentense, grnn_cdnn_fw_out,
            rv_batch2seq.out); // reuse rv_batch2seq.out
        // fw_seq_last_pooling
        fw_seq_last_pooling.infer(ctx, sentense, fw_batch2seq.out,
            fw_pool_out_l3);
        // concat_2in
        concat_2in.infer(ctx, sentense.seqlen_sum,
            fw_batch2seq.out, seq_reverse.out,
            concat_2in_out_l3);
        // concat_3in
        concat_3in.infer(ctx, sentense.seqlen_sum,
            bid_emb_ew_fw_out, fw_batch2seq.out, seq_reverse.out,
            concat_3in_out_l3);
        return l3_manager.max_used_l3_size;
    }
};

class QPcqEmb {
    float* table;
    int table_len;
    int emb_dim;
    // hbm_buffer
    float* out_q_hbm;
    float* out_pcq_hbm;
public:
    float* out_q; // [q.cap_l, emb_dim]
    float* out_pcq; // [pcq.cap_l, emb_dim]

    void init(lite::Tensor* table_tensor, int UB_batch, int UB_seqlen) {
        table = table_tensor->mutable_data<float>();
        table_len = table_tensor->dims()[0];
        emb_dim = table_tensor->dims()[1];
        // hbm_buffer
        int max_cap_len = UB_batch * UB_seqlen;
        xpu_malloc((void**)(&out_q_hbm), max_cap_len * emb_dim * sizeof(float));
        xpu_malloc((void**)(&out_pcq_hbm), max_cap_len * emb_dim * sizeof(float));
    }

    int out_q_size(const IdInfo& sentense) {
        return sentense.seqlen_sum * emb_dim * sizeof(float);
    }

    int out_pcq_size(const IdInfo& sentense) {
        return sentense.seqlen_sum * emb_dim * sizeof(float);
    }

    int total_result_size(const IdInfo& q_sentense, const IdInfo& pcq_sentense) {
        return out_q_size(q_sentense) + out_pcq_size(pcq_sentense);
    }

    void infer(api::Context* ctx,
            const IdInfo& q_id, const IdInfo& pcq_id,
            float* out_q_l3 = nullptr, float* out_pcq_l3 = nullptr) {
        out_q = (out_q_l3 == nullptr) ? out_q_hbm : out_q_l3;
        out_pcq = (out_pcq_l3 == nullptr) ? out_pcq_hbm : out_pcq_l3;
        api::embedding<float, int64_t>(ctx,
            q_id.lod_32_cpu.back(),
            q_id.id0_64,
            emb_dim,
            table,
            out_q,
            table_len - 2);
        api::embedding<float, int64_t>(ctx,
            pcq_id.lod_32_cpu.back(),
            pcq_id.id0_64,
            emb_dim,
            table,
            out_pcq,
            table_len - 2);
    }
};

class MatchConvCDNN {
    int dim_t;
    int dim_in;
    int out_channel;
public:
    FindMax find_max;
    FC fc;
    MatchMatrixTensor match;
    SearchVarConv conv;

    void init(
            lite::Tensor* match_weight, float match_weight_max,
            lite::Tensor* conv_weight, float conv_weight_max,
            int _dim_t, int _dim_in, int _out_channel,
            int UB_batch, int UB_seqlen) {
        dim_t = _dim_t;
        dim_in = _dim_in;
        out_channel = _out_channel;
        //
        find_max.init();
        //
        fc.init(UB_batch * UB_seqlen, dim_t * dim_in, dim_in,
            match_weight->mutable_data<int16_t>(), match_weight_max);
        //
        match.init(dim_t, dim_in, UB_batch, UB_seqlen);
        //
        conv.init(conv_weight->mutable_data<int16_t>(), conv_weight_max,
            dim_t, out_channel, UB_batch, UB_seqlen);
    }

    int infer(api::Context* ctx, const IdInfo& left_sentense, const IdInfo& right_sentense,
            const float* left, const float* right,
            char* l3_workspace = nullptr, int l3_size = 0,
            float* match_out_l3 = nullptr, float* conv_out_l3 = nullptr) {
        L3Manager l3_manager(l3_workspace, l3_size);
        // find max
        int workspace_need_size = find_max.total_result_size();
        find_max.infer(ctx, left, left_sentense.seqlen_sum * dim_in,
            l3_manager.malloc(workspace_need_size));
        // fc
        workspace_need_size = fc.total_result_size(left_sentense.seqlen_sum);
        float* fc_out_max_l3 = l3_manager.malloc(workspace_need_size);
        float* fc_out_l3 = (fc_out_max_l3 == nullptr) ? nullptr : (fc_out_max_l3 + 4);
        fc.infer(ctx, left, find_max.out, left_sentense.seqlen_sum,
            fc_out_l3, fc_out_max_l3);
        // match
        match.infer(ctx, left_sentense, right_sentense,
            fc.out, fc.out_max, right,
            match_out_l3);
        // conv
        conv.infer(ctx, left_sentense, right_sentense,
            match.out, conv_out_l3);
        return l3_manager.max_used_l3_size;
    }
};

class Attention {
    int dim;
    float scale0;
    float scale1;
    //
    FindMax input_find_max;
    FC seqfc;
    float* hbm_workspace; // require: cap_l * dim + seqlen_square_sum
    // seqfc_out: [cap_l, dim]
    // batchgemm0_out: [seqlen_square_sum]
    // seq_softmax_out: [seqlen_square_sum], reuse of batchgemm0_out
    // batchgemm1_out: [cap_l, dim], reuse of seqfc_out
    float* out_hbm;
public:
    float* out; // [batch, dim]

    void init(lite::Tensor* fc_w, float fc_w_max, lite::Tensor* fc_b,
            int _dim, int UB_batch, int UB_seqlen) {
        dim = _dim;
        scale0 = 0.0883883461356163f;
        scale1 = 1.0f;
        input_find_max.init();
        seqfc.init(UB_batch * UB_seqlen, dim, dim,
            fc_w->mutable_data<int16_t>(), fc_w_max, fc_b->mutable_data<float>(),
            api::Activation_t::LINEAR);
        xpu_malloc((void**)(&hbm_workspace),
            (UB_batch * (UB_seqlen * dim + UB_seqlen * UB_seqlen)) * sizeof(float));
        xpu_malloc((void**)(&out_hbm), UB_batch * dim * sizeof(float));
    }

    int out_size(const IdInfo& sentense) {
        return sentense.batch * dim * sizeof(float);
    }

    int total_result_size(const IdInfo& sentense) {
        return out_size(sentense);
    }

    int infer(api::Context* ctx, const IdInfo& sentense,
            const float* input,
            char* l3_workspace = nullptr, int l3_size = 0,
            float* out_l3 = nullptr) {
        int max_used_l3_size = 0;
        int batch = sentense.batch;
        int cap_l = sentense.seqlen_sum;
        int max_width = sentense.seqlen_max;
        int* lod_32 = sentense.lod_32;

        int workspace_need_size = (cap_l * dim + sentense.seqlen_square_sum) * sizeof(float);
        float* workspace = hbm_workspace;
        if ((l3_workspace != nullptr) && (l3_size >= workspace_need_size)) {
            workspace = (float*)l3_workspace;
            //l3_workspace += workspace_need_size;
            //l3_size -= workspace_need_size;
            max_used_l3_size += workspace_need_size;
        }
        float* seqfc_out = workspace;
        float* batchgemm0_out = seqfc_out + cap_l * dim;
        float* seq_softmax_out = batchgemm0_out;
        float* batchgemm1_out = seqfc_out;

        input_find_max.infer(ctx, input, cap_l * dim);
        seqfc.infer(ctx, input, input_find_max.out, cap_l, seqfc_out);
        api::search_noaligned_mat_mul(ctx, 0, 1,
                batch, lod_32, max_width, dim,
                scale0, input, seqfc_out, batchgemm0_out);
        api::search_seq_softmax(ctx, batchgemm0_out, seq_softmax_out, lod_32, batch, max_width);
        api::search_noaligned_mat_mul(ctx, 0, 0,
                batch, lod_32, max_width, dim,
                scale1, seq_softmax_out, input, batchgemm1_out);
        out = (out_l3 == nullptr) ? out_hbm : out_l3;
        api::sequence_pooling_forward(ctx, api::Pooling_t::MAX_WITHOUT_INDEX,
                batch, lod_32, dim,
                batchgemm1_out, nullptr, out);
        return max_used_l3_size;
    }
};

class MergeAllBeginV1 {
public:
    SeqConcat q_pa_seq_concat;
    TopkAvgPooling q_pa_topk_avg_pooling;
    SeqConcat q_pt_seq_concat;
    TopkAvgPooling q_pt_topk_avg_pooling;
    Concat2In concat_2in; // q_pt, q_pa
    Seq2Batch fw_seq2batch;
    SeqReverse seq_reverse;
    Seq2Batch rv_seq2batch;

    void init(
            int _q_pa_dim_t, int _q_pa_out_channel, std::vector<int> _q_pa_topks,
            int _q_pt_dim_t, int _q_pt_out_channel, std::vector<int> _q_pt_topks,
            int UB_batch, int UB_seqlen) {
        int max_cap_l = UB_batch * UB_seqlen;
        q_pa_seq_concat.init(_q_pa_dim_t, _q_pa_out_channel, UB_batch, UB_seqlen);
        q_pa_topk_avg_pooling.init(_q_pa_dim_t, _q_pa_out_channel,
            UB_batch, UB_seqlen, _q_pa_topks);
        q_pt_seq_concat.init(_q_pt_dim_t, _q_pt_out_channel, UB_batch, UB_seqlen);
        q_pt_topk_avg_pooling.init(_q_pt_dim_t, _q_pt_out_channel,
            UB_batch, UB_seqlen, _q_pt_topks);
        int q_pa_channel = (_q_pa_dim_t + _q_pa_out_channel) * _q_pa_topks.size();
        assert(q_pa_channel == 32);
        int q_pt_channel = (_q_pt_dim_t + _q_pt_out_channel) * _q_pt_topks.size();
        assert(q_pt_channel == 30);
        concat_2in.init(max_cap_l, q_pt_channel, q_pa_channel);
        int bi_grnn_dim_in = q_pt_channel + q_pa_channel;
        fw_seq2batch.init(bi_grnn_dim_in, max_cap_l);
        seq_reverse.init(bi_grnn_dim_in, max_cap_l);
        rv_seq2batch.init(bi_grnn_dim_in, max_cap_l);
    }

    void infer(api::Context* ctx, const IdInfo& q_sentense,
            const IdInfo& pa_sentense,
            const float* q_pa_match_out, const float* q_pa_varconv_out,
            const IdInfo& pt_sentense,
            const float* q_pt_match_out, const float* q_pt_varconv_out,
            char* l3_workspace = nullptr, int l3_size = 0,
            float* fw_seq2batch_out_l3 = nullptr, float* rv_seq2batch_out_l3 = nullptr) {
        L3BottomTopManager l3_manager(l3_workspace, l3_size);
        // q_pa_seq_concat bottom
        int workspace_need_size = q_pa_seq_concat.total_result_size(q_sentense, pa_sentense);
        q_pa_seq_concat.infer(ctx, q_sentense, pa_sentense,
            q_pa_match_out, q_pa_varconv_out,
            l3_manager.malloc_bottom(workspace_need_size));
        // q_pa_topk_avg_pooling top
        workspace_need_size = q_pa_topk_avg_pooling.total_result_size(q_sentense, pa_sentense);
        q_pa_topk_avg_pooling.infer(ctx, q_sentense, pa_sentense,
            q_pa_seq_concat.out,
            l3_manager.malloc_top(workspace_need_size));
        // free bottom
        l3_manager.free_bottom();
        // q_pt_seq_concat bottom
        workspace_need_size = q_pt_seq_concat.total_result_size(q_sentense, pt_sentense);
        q_pt_seq_concat.infer(ctx, q_sentense, pt_sentense,
            q_pt_match_out, q_pt_varconv_out,
            l3_manager.malloc_bottom(workspace_need_size));
        // q_pt_topk_avg_pooling top
        workspace_need_size = q_pt_topk_avg_pooling.total_result_size(q_sentense, pt_sentense);
        q_pt_topk_avg_pooling.infer(ctx, q_sentense, pt_sentense,
            q_pt_seq_concat.out,
            l3_manager.malloc_top(workspace_need_size));
        // free bottom
        l3_manager.free_bottom();
        // concat_2in bottom
        workspace_need_size = concat_2in.total_result_size(q_sentense.seqlen_sum);
        concat_2in.infer(ctx, q_sentense.seqlen_sum,
            q_pt_topk_avg_pooling.out,
            q_pa_topk_avg_pooling.out,
            l3_manager.malloc_bottom(workspace_need_size)); // q_pt, q_pa
        // free top
        l3_manager.free_top();
        // fw_seq2batch
        fw_seq2batch.infer(ctx, q_sentense, concat_2in.out,
            fw_seq2batch_out_l3);
        // seq_reverse top
        workspace_need_size = seq_reverse.total_result_size(q_sentense);
        seq_reverse.infer(ctx, q_sentense, concat_2in.out,
            l3_manager.malloc_top(workspace_need_size));
        // rv_seq2batch
        rv_seq2batch.infer(ctx, q_sentense, seq_reverse.out,
            rv_seq2batch_out_l3);
    }
};

class MergeAllBeginV2 {
public:
    SeqConcat q_pcq_seq_concat;
    TopkAvgPooling q_pcq_topk_avg_pooling;
    SeqConcat q_pa_seq_concat;
    TopkAvgPooling q_pa_topk_avg_pooling;
    SeqConcat q_pt_seq_concat;
    TopkAvgPooling q_pt_topk_avg_pooling;
    Concat3In concat_3in; // q_pt, q_pa, q_pcq
    Seq2Batch fw_seq2batch;
    SeqReverse seq_reverse;
    Seq2Batch rv_seq2batch;

    void init(
            int _q_pcq_dim_t, int _q_pcq_out_channel, std::vector<int> _q_pcq_topks,
            int _q_pa_dim_t, int _q_pa_out_channel, std::vector<int> _q_pa_topks,
            int _q_pt_dim_t, int _q_pt_out_channel, std::vector<int> _q_pt_topks,
            int UB_batch, int UB_seqlen) {
        int max_cap_l = UB_batch * UB_seqlen;
        q_pcq_seq_concat.init(_q_pcq_dim_t, _q_pcq_out_channel, UB_batch, UB_seqlen);
        q_pcq_topk_avg_pooling.init(_q_pcq_dim_t, _q_pcq_out_channel,
            UB_batch, UB_seqlen, _q_pcq_topks);
        q_pa_seq_concat.init(_q_pa_dim_t, _q_pa_out_channel, UB_batch, UB_seqlen);
        q_pa_topk_avg_pooling.init(_q_pa_dim_t, _q_pa_out_channel,
            UB_batch, UB_seqlen, _q_pa_topks);
        q_pt_seq_concat.init(_q_pt_dim_t, _q_pt_out_channel, UB_batch, UB_seqlen);
        q_pt_topk_avg_pooling.init(_q_pt_dim_t, _q_pt_out_channel,
            UB_batch, UB_seqlen, _q_pt_topks);
        int q_pcq_channel = (_q_pcq_dim_t + _q_pcq_out_channel) * _q_pcq_topks.size();
        assert(q_pcq_channel == 72);
        int q_pa_channel = (_q_pa_dim_t + _q_pa_out_channel) * _q_pa_topks.size();
        assert(q_pa_channel == 32);
        int q_pt_channel = (_q_pt_dim_t + _q_pt_out_channel) * _q_pt_topks.size();
        assert(q_pt_channel == 30);
        concat_3in.init(max_cap_l, q_pt_channel, q_pa_channel, q_pcq_channel);
        int bi_grnn_dim_in = q_pt_channel + q_pa_channel + q_pcq_channel;
        fw_seq2batch.init(bi_grnn_dim_in, max_cap_l);
        seq_reverse.init(bi_grnn_dim_in, max_cap_l);
        rv_seq2batch.init(bi_grnn_dim_in, max_cap_l);
    }

    void infer(api::Context* ctx, const IdInfo& q_sentense,
            const IdInfo& pcq_sentense,
            const float* q_pcq_match_out, const float* q_pcq_varconv_out,
            const IdInfo& pa_sentense,
            const float* q_pa_match_out, const float* q_pa_varconv_out,
            const IdInfo& pt_sentense,
            const float* q_pt_match_out, const float* q_pt_varconv_out,
            char* l3_workspace = nullptr, int l3_size = 0,
            float* fw_seq2batch_out_l3 = nullptr, float* rv_seq2batch_out_l3 = nullptr) {
        L3BottomTopManager l3_manager(l3_workspace, l3_size);
        // q_pcq_seq_concat bottom
        int workspace_need_size = q_pcq_seq_concat.total_result_size(q_sentense, pcq_sentense);
        q_pcq_seq_concat.infer(ctx, q_sentense, pcq_sentense,
            q_pcq_match_out, q_pcq_varconv_out,
            l3_manager.malloc_bottom(workspace_need_size));
        // q_pcq_topk_avg_pooling top
        workspace_need_size = q_pcq_topk_avg_pooling.total_result_size(q_sentense, pcq_sentense);
        q_pcq_topk_avg_pooling.infer(ctx, q_sentense, pcq_sentense,
            q_pcq_seq_concat.out,
            l3_manager.malloc_top(workspace_need_size));
        // free bottom
        l3_manager.free_bottom();
        // q_pa_seq_concat bottom
        workspace_need_size = q_pa_seq_concat.total_result_size(q_sentense, pa_sentense);
        q_pa_seq_concat.infer(ctx, q_sentense, pa_sentense,
            q_pa_match_out, q_pa_varconv_out,
            l3_manager.malloc_bottom(workspace_need_size));
        // q_pa_topk_avg_pooling top
        workspace_need_size = q_pa_topk_avg_pooling.total_result_size(q_sentense, pa_sentense);
        q_pa_topk_avg_pooling.infer(ctx, q_sentense, pa_sentense,
            q_pa_seq_concat.out,
            l3_manager.malloc_top(workspace_need_size));
        // free bottom
        l3_manager.free_bottom();
        // q_pt_seq_concat bottom
        workspace_need_size = q_pt_seq_concat.total_result_size(q_sentense, pt_sentense);
        q_pt_seq_concat.infer(ctx, q_sentense, pt_sentense,
            q_pt_match_out, q_pt_varconv_out,
            l3_manager.malloc_bottom(workspace_need_size));
        // q_pt_topk_avg_pooling top
        workspace_need_size = q_pt_topk_avg_pooling.total_result_size(q_sentense, pt_sentense);
        q_pt_topk_avg_pooling.infer(ctx, q_sentense, pt_sentense,
            q_pt_seq_concat.out,
            l3_manager.malloc_top(workspace_need_size));
        // free bottom
        l3_manager.free_bottom();
        // concat_3in bottom
        workspace_need_size = concat_3in.total_result_size(q_sentense.seqlen_sum);
        concat_3in.infer(ctx, q_sentense.seqlen_sum,
            q_pt_topk_avg_pooling.out,
            q_pa_topk_avg_pooling.out,
            q_pcq_topk_avg_pooling.out,
            l3_manager.malloc_bottom(workspace_need_size)); // q_pt, q_pa, q_pcq
        // free top
        l3_manager.free_top();
        // fw_seq2batch
        fw_seq2batch.infer(ctx, q_sentense, concat_3in.out,
            fw_seq2batch_out_l3);
        // seq_reverse top
        workspace_need_size = seq_reverse.total_result_size(q_sentense);
        seq_reverse.infer(ctx, q_sentense, concat_3in.out,
            l3_manager.malloc_top(workspace_need_size));
        // rv_seq2batch
        rv_seq2batch.infer(ctx, q_sentense, seq_reverse.out,
            rv_seq2batch_out_l3);
    }
};

class MergeAllEnd {
    const int fc_0_k = 1152;
    const int fc_0_n = 512;
    const int fc_1_k = 640;
    const int fc_1_n = 320;
    const int fc_2_k = 320;
    const int fc_2_n = 1;
public:
    // bottom
    BiGrnnCDNN bi_grnn_cdnn;
    // top
    Batch2Seq fw_batch2seq;
    Batch2Seq rv_batch2seq;
    // bottom
    SeqLastPooling fw_seq_last_pooling;
    SeqLastPooling rv_seq_last_pooling;
    // top
    Concat7In concat_7in;
    // bottom
    FindMax find_max_0;
    FC fc_0;
    // top
    Concat3In concat_3in;
    // bottom
    FindMax find_max_1;
    FC fc_1;
    FC fc_2;

    void init(
            lite::Tensor* grnn_fw_wh, std::vector<float>& grnn_fw_wh_maxs,
            lite::Tensor* grnn_fw_wi, std::vector<float>& grnn_fw_wi_maxs,
            lite::Tensor* grnn_rv_wh, std::vector<float>& grnn_rv_wh_maxs,
            lite::Tensor* grnn_rv_wi, std::vector<float>& grnn_rv_wi_maxs,
            int _cap_e, int _cap_h,
            lite::Tensor* fc0_w, float fc0_w_max, lite::Tensor* fc0_b,
            lite::Tensor* fc1_w, float fc1_w_max, lite::Tensor* fc1_b,
            lite::Tensor* fc2_w, float fc2_w_max, lite::Tensor* fc2_b,
            int UB_batch, int UB_seqlen) {
        int max_cap_l = UB_batch * UB_seqlen;
        bi_grnn_cdnn.init(
            grnn_fw_wh, grnn_fw_wh_maxs,
            grnn_fw_wi, grnn_fw_wi_maxs,
            grnn_rv_wh, grnn_rv_wh_maxs,
            grnn_rv_wi, grnn_rv_wi_maxs,
            _cap_e, _cap_h, UB_batch, UB_seqlen);
        fw_batch2seq.init(_cap_h, max_cap_l);
        rv_batch2seq.init(_cap_h, max_cap_l);
        fw_seq_last_pooling.init(_cap_h, UB_batch);
        rv_seq_last_pooling.init(_cap_h, UB_batch);
        concat_7in.init(UB_batch, 128, 128, 256, 128, 128, 256, 128);
        find_max_0.init();
        fc_0.init(UB_batch, fc_0_n, fc_0_k,
            fc0_w->mutable_data<int16_t>(), fc0_w_max, fc0_b->mutable_data<float>(),
            api::Activation_t::RELU);
        concat_3in.init(UB_batch, 64, 64, 512);
        find_max_1.init();
        fc_1.init(UB_batch, fc_1_n, fc_1_k,
            fc1_w->mutable_data<int16_t>(), fc1_w_max, fc1_b->mutable_data<float>(),
            api::Activation_t::RELU);
        fc_2.init(UB_batch, fc_2_n, fc_2_k,
            fc2_w->mutable_data<int16_t>(), fc2_w_max, fc2_b->mutable_data<float>(),
            api::Activation_t::LINEAR);
    }

    void infer(api::Context* ctx, const IdInfo& sentense,
            const float* bi_grnn_fw_in, const float* bi_grnn_rv_in,
            const float* q_fw_pool, const float* q_rv_pool, const float* q_att_out,
            const float* pt_fw_pool, const float* pt_rv_pool, const float* pt_att_out,
            const float* pa_att,
            char* l3_workspace = nullptr, int l3_size = 0,
            float* out_l3 = nullptr) {
        L3BottomTopManager l3_manager(l3_workspace, l3_size);
        // bi_grnn_cdnn bottom
        int workspace_need_size = bi_grnn_cdnn.total_result_size(sentense);
        float* fw_out_l3 = l3_manager.malloc_bottom(workspace_need_size);
        float* rv_out_l3 = (fw_out_l3 == nullptr) ? nullptr :
            (float*)((char*)fw_out_l3 + bi_grnn_cdnn.bi_grnn.fw_out_size(sentense));
        bi_grnn_cdnn.infer(ctx, sentense,
            bi_grnn_fw_in, bi_grnn_rv_in,
            l3_manager.unused_workspace(), l3_manager.unused_size(),
            fw_out_l3, rv_out_l3);
        // fw_batch2seq top
        workspace_need_size = fw_batch2seq.total_result_size(sentense);
        fw_batch2seq.infer(ctx, sentense, bi_grnn_cdnn.bi_grnn.fw_out,
            l3_manager.malloc_top(workspace_need_size));
        // rv_batch2seq top
        workspace_need_size = rv_batch2seq.total_result_size(sentense);
        rv_batch2seq.infer(ctx, sentense, bi_grnn_cdnn.bi_grnn.rv_out,
            l3_manager.malloc_top(workspace_need_size));
        // release all bottom
        l3_manager.free_bottom();
        // fw_seq_last_pooling bottom
        workspace_need_size = fw_seq_last_pooling.total_result_size(sentense);
        fw_seq_last_pooling.infer(ctx, sentense, fw_batch2seq.out,
            l3_manager.malloc_bottom(workspace_need_size));
        // rv_seq_last_pooling bottom
        workspace_need_size = rv_seq_last_pooling.total_result_size(sentense);
        rv_seq_last_pooling.infer(ctx, sentense, rv_batch2seq.out,
            l3_manager.malloc_bottom(workspace_need_size));
        // release all top
        l3_manager.free_top();
        // concat_7in top
        workspace_need_size = concat_7in.total_result_size(sentense.batch);
        concat_7in.infer(ctx, sentense.batch,
            q_fw_pool, q_rv_pool, q_att_out,
            pt_fw_pool, pt_rv_pool, pt_att_out,
            pa_att,
            l3_manager.malloc_top(workspace_need_size));
        // find_max_0 bottom
        workspace_need_size = find_max_0.total_result_size();
        find_max_0.infer(ctx, concat_7in.out, sentense.batch * fc_0_k,
            l3_manager.malloc_bottom(workspace_need_size));
        // fc_0 bottom
        workspace_need_size = fc_0.total_result_size(sentense.batch);
        float* fc_0_out_max_l3 = l3_manager.malloc_bottom(workspace_need_size);
        float* fc_0_out_l3 = (fc_0_out_max_l3 == nullptr) ? nullptr : (fc_0_out_max_l3 + 4);
        fc_0.infer(ctx, concat_7in.out, find_max_0.out, sentense.batch,
            fc_0_out_l3, fc_0_out_max_l3);
        // release all top
        l3_manager.free_top();
        // concat_3in top
        workspace_need_size = concat_3in.total_result_size(sentense.batch);
        concat_3in.infer(ctx, sentense.batch,
            fw_seq_last_pooling.out, rv_seq_last_pooling.out, fc_0.out,
            l3_manager.malloc_bottom(workspace_need_size));
        // release all bottom
        l3_manager.free_bottom();
        // find_max_1 bottom
        workspace_need_size = find_max_1.total_result_size();
        find_max_1.infer(ctx, concat_3in.out, sentense.batch * fc_1_k,
            l3_manager.malloc_bottom(workspace_need_size));
        // fc_1 bottom
        workspace_need_size = fc_1.total_result_size(sentense.batch);
        float* fc_1_out_max_l3 = l3_manager.malloc_bottom(workspace_need_size);
        float* fc_1_out_l3 = (fc_1_out_max_l3 == nullptr) ? nullptr : (fc_1_out_max_l3 + 4);
        fc_1.infer(ctx, concat_3in.out, find_max_1.out, sentense.batch,
            fc_1_out_l3, fc_1_out_max_l3);
        // fc_2 -> out_l3
        fc_2.infer(ctx, fc_1.out, fc_1.out_max, sentense.batch,
            out_l3);
    }
};

class XPUMmdnnMultiStreamV1Compute
        : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
public:
    using param_t = operators::XPUMmdnnMultiStreamV1Param;

    void PrepareForRun() override;

    void Run() override;

private:
    int UB_batch = 20;      //  upper bound of batch
    int UB_seqlen = 512;    //  upper bound of seqlen
    api::Context* ctx_1{nullptr};
    api::Context* ctx_2{nullptr};
    IdInfo q_id;
    IdInfo pt_id;
    IdInfo pa_id;
    BiGrnnBegin q_bi_grnn_begin;
    BiGrnnCDNN q_bi_grnn_cdnn;
    BiGrnnEnd q_bi_grnn_end;
    BiGrnnBegin pt_bi_grnn_begin;
    BiGrnnCDNN pt_bi_grnn_cdnn;
    BiGrnnEnd pt_bi_grnn_end;
    EmbEw pa_emb_ew;
    MatchConvCDNN q_pa_match_conv_cdnn;
    MatchConvCDNN q_pt_match_conv_cdnn;
    Attention q_att;
    Attention pt_att;
    Attention pa_att;
    MergeAllBeginV1 merge_all_begin;
    MergeAllEnd merge_all_end;
};

void XPUMmdnnMultiStreamV1Compute::PrepareForRun() {
    auto& param = this->Param<param_t>();

    XPUStream stream_1;
    int ret = xpu_stream_create(&stream_1);
    if (ret) {
        std::cout << "stream_1 create fail!" << std::endl;
        return;
    }
    ctx_1 = api::create_context();
    api::set_workspace_l3_size(ctx_1, 0);
    ctx_1->xpu_stream = stream_1;

    XPUStream stream_2;
    ret = xpu_stream_create(&stream_2);
    if (ret) {
        std::cout << "stream_2 create fail!" << std::endl;
        return;
    }
    ctx_2 = api::create_context();
    api::set_workspace_l3_size(ctx_2, 0);
    ctx_2->xpu_stream = stream_2;

    const int emb_dim = 128;
    int _cap_e = emb_dim;
    int _cap_h = 128;
    int q_pa_dim_t = 3;
    int q_pa_out_channel = 5;
    std::vector<int> q_pa_topks = {1, 2, 5, 12};
    int q_pt_dim_t = 5;
    int q_pt_out_channel = 5;
    std::vector<int> q_pt_topks = {1, 2, 4};

    q_id.init(UB_batch, UB_seqlen);
    pt_id.init(UB_batch, UB_seqlen);
    pa_id.init(UB_batch, UB_seqlen);

    q_bi_grnn_begin.init(param.emb_tbl, UB_batch, UB_seqlen);
    q_bi_grnn_cdnn.init(
        param.q_bid_emb_grnn_att_grnn_fw_wh,
        param.q_bid_emb_grnn_att_grnn_fw_wh_maxs,
        param.q_bid_emb_grnn_att_grnn_fw_wi,
        param.q_bid_emb_grnn_att_grnn_fw_wi_maxs,
        param.q_bid_emb_grnn_att_grnn_rv_wh,
        param.q_bid_emb_grnn_att_grnn_rv_wh_maxs,
        param.q_bid_emb_grnn_att_grnn_rv_wi,
        param.q_bid_emb_grnn_att_grnn_rv_wi_maxs,
        _cap_e, _cap_h, UB_batch, UB_seqlen);
    q_bi_grnn_end.init(_cap_e, _cap_h, UB_batch, UB_seqlen);

    pt_bi_grnn_begin.init(param.emb_tbl, UB_batch, UB_seqlen);
    pt_bi_grnn_cdnn.init(
        param.pt_bid_emb_grnn_att_grnn_fw_wh,
        param.pt_bid_emb_grnn_att_grnn_fw_wh_maxs,
        param.pt_bid_emb_grnn_att_grnn_fw_wi,
        param.pt_bid_emb_grnn_att_grnn_fw_wi_maxs,
        param.pt_bid_emb_grnn_att_grnn_rv_wh,
        param.pt_bid_emb_grnn_att_grnn_rv_wh_maxs,
        param.pt_bid_emb_grnn_att_grnn_rv_wi,
        param.pt_bid_emb_grnn_att_grnn_rv_wi_maxs,
        _cap_e, _cap_h, UB_batch, UB_seqlen);
    pt_bi_grnn_end.init(_cap_e, _cap_h, UB_batch, UB_seqlen);

    pa_emb_ew.init(param.emb_tbl, UB_batch, UB_seqlen);
    q_pa_match_conv_cdnn.init(
        param.q_pa_match_conv_topk_input_w,
        param.q_pa_match_conv_topk_input_w_max,
        param.q_pa_match_conv_topk_conv_w,
        param.q_pa_match_conv_topk_conv_w_max,
        q_pa_dim_t, emb_dim, q_pa_out_channel,
        UB_batch, UB_seqlen);
    q_pt_match_conv_cdnn.init(
        param.q_pt_match_conv_topk_input_w,
        param.q_pt_match_conv_topk_input_w_max,
        param.q_pt_match_conv_topk_conv_w,
        param.q_pt_match_conv_topk_conv_w_max,
        q_pt_dim_t, _cap_h * 3, q_pt_out_channel,
        UB_batch, UB_seqlen);

    q_att.init(
        param.q_bid_emb_grnn_att_att_fc_w,
        param.q_bid_emb_grnn_att_att_fc_w_max,
        param.q_bid_emb_grnn_att_att_fc_b,
        2 * _cap_h, UB_batch, UB_seqlen);
    pt_att.init(
        param.pt_bid_emb_grnn_att_att_fc_w,
        param.pt_bid_emb_grnn_att_att_fc_w_max,
        param.pt_bid_emb_grnn_att_att_fc_b,
        2 * _cap_h, UB_batch, UB_seqlen);
    pa_att.init(
        param.pa_bid_emb_att_att_fc_w,
        param.pa_bid_emb_att_att_fc_w_max,
        param.pa_bid_emb_att_att_fc_b,
        emb_dim, UB_batch, UB_seqlen);

    merge_all_begin.init(
        q_pa_dim_t, q_pa_out_channel, q_pa_topks,
        q_pt_dim_t, q_pt_out_channel, q_pt_topks,
        UB_batch, UB_seqlen);

    merge_all_end.init(
        param.merge_all_grnn_fw_wh,
        param.merge_all_grnn_fw_wh_maxs,
        param.merge_all_grnn_fw_wi,
        param.merge_all_grnn_fw_wi_maxs,
        param.merge_all_grnn_rv_wh,
        param.merge_all_grnn_rv_wh_maxs,
        param.merge_all_grnn_rv_wi,
        param.merge_all_grnn_rv_wi_maxs,
        (30 + 32), 64,
        param.merge_all_fc0_w,
        param.merge_all_fc0_w_max,
        param.merge_all_fc0_b,
        param.merge_all_fc1_w,
        param.merge_all_fc1_w_max,
        param.merge_all_fc1_b,
        param.merge_all_fc2_w,
        param.merge_all_fc2_w_max,
        param.merge_all_fc2_b,
        UB_batch, UB_seqlen);

    xpu_disp_set_single_mode();
}

void XPUMmdnnMultiStreamV1Compute::Run() {
    auto& param = this->Param<param_t>();
    int batch = param.q_basic->lod()[0].size() - 1;
    if (batch > UB_batch) {
        LOG(FATAL) << "Batch of MMDNN should not be larger than " << UB_batch << std::endl;
    }

    auto& ctx = this->ctx_->As<XPUContext>();
    auto* xpu_ctx = ctx.GetRawContext();
    char* l3_buffer_base = (char*)(xpu_ctx->workspace_l3_ptr);
    int total_l3_size = xpu_ctx->workspace_l3_size;
    q_id.get_xpu_buffer_l3(l3_buffer_base, total_l3_size);
    pt_id.get_xpu_buffer_l3(l3_buffer_base, total_l3_size);
    pa_id.get_xpu_buffer_l3(l3_buffer_base, total_l3_size);
    L3BottomTopManager l3_manager(l3_buffer_base, total_l3_size);

    int workspace_need_size = 0;
    float* workspace_base_addr = nullptr;
    char* l3_buffer = nullptr;
    int l3_size = 0;
    int stream_used_l3_size = 0;

    // stage-0
    q_id.update(param.q_basic, param.q_bigram0);
    pt_id.update(param.pt_basic, param.pt_bigram0);
    pa_id.update(param.pa_basic, param.pa_bigram0);
    // stage-1 prepare
    // l3_malloc bottom
    // q_bi_grnn_begin.fw_seq2batch.out
    // q_bi_grnn_begin.rv_seq2batch.out
    int q_bi_grnn_begin_fw_seq2batch_out_size =
        q_bi_grnn_begin.fw_seq2batch.out_size(q_id);
    int q_bi_grnn_begin_rv_seq2batch_out_size =
        q_bi_grnn_begin.rv_seq2batch.out_size(q_id);
    workspace_need_size = q_bi_grnn_begin_fw_seq2batch_out_size
        + q_bi_grnn_begin_rv_seq2batch_out_size;
    workspace_base_addr = l3_manager.malloc_bottom(workspace_need_size);
    l3_buffer = l3_manager.unused_workspace();
    l3_size = l3_manager.unused_size();
    float* q_bi_grnn_begin_fw_seq2batch_out_l3 = nullptr;
    float* q_bi_grnn_begin_rv_seq2batch_out_l3 = nullptr;
    if (workspace_base_addr != nullptr) {
        q_bi_grnn_begin_fw_seq2batch_out_l3 = workspace_base_addr;
        q_bi_grnn_begin_rv_seq2batch_out_l3 = (float*)
            ((char*)q_bi_grnn_begin_fw_seq2batch_out_l3
            + q_bi_grnn_begin_fw_seq2batch_out_size);
    }
    // stage-1
    // ctx_1 only
    q_bi_grnn_begin.infer(ctx_1, q_id,
        l3_buffer, l3_size,
        nullptr, // bid_emb_ew_out_fw_l3
        q_bi_grnn_begin_fw_seq2batch_out_l3,
        q_bi_grnn_begin_rv_seq2batch_out_l3);
    // stage-2 prepare
    // l3_malloc top
    // q_bi_grnn_cdnn.bi_grnn.fw_out
    // q_bi_grnn_cdnn.bi_grnn.rv_out
    // pt_bi_grnn_begin.fw_seq2batch.out
    // pt_bi_grnn_begin.rv_seq2batch.out
    int q_bi_grnn_cdnn_bi_grnn_fw_out_size =
        q_bi_grnn_cdnn.bi_grnn.fw_out_size(q_id);
    int q_bi_grnn_cdnn_bi_grnn_rv_out_size =
        q_bi_grnn_cdnn.bi_grnn.rv_out_size(q_id);
    int pt_bi_grnn_begin_fw_seq2batch_out_size =
        pt_bi_grnn_begin.fw_seq2batch.out_size(pt_id);
    int pt_bi_grnn_begin_rv_seq2batch_out_size =
        pt_bi_grnn_begin.rv_seq2batch.out_size(pt_id);
    workspace_need_size = q_bi_grnn_cdnn_bi_grnn_fw_out_size
        + q_bi_grnn_cdnn_bi_grnn_rv_out_size
        + pt_bi_grnn_begin_fw_seq2batch_out_size
        + pt_bi_grnn_begin_rv_seq2batch_out_size;
    workspace_base_addr = l3_manager.malloc_top(workspace_need_size);
    l3_buffer = l3_manager.unused_workspace();
    l3_size = l3_manager.unused_size();
    float* q_bi_grnn_cdnn_bi_grnn_fw_out_l3 = nullptr;
    float* q_bi_grnn_cdnn_bi_grnn_rv_out_l3 = nullptr;
    float* pt_bi_grnn_begin_fw_seq2batch_out_l3 = nullptr;
    float* pt_bi_grnn_begin_rv_seq2batch_out_l3 = nullptr;
    if (workspace_base_addr != nullptr) {
        q_bi_grnn_cdnn_bi_grnn_fw_out_l3 = workspace_base_addr;
        q_bi_grnn_cdnn_bi_grnn_rv_out_l3 = (float*)
            ((char*)q_bi_grnn_cdnn_bi_grnn_fw_out_l3
            + q_bi_grnn_cdnn_bi_grnn_fw_out_size);
        pt_bi_grnn_begin_fw_seq2batch_out_l3 = (float*)
            ((char*)q_bi_grnn_cdnn_bi_grnn_rv_out_l3
            + q_bi_grnn_cdnn_bi_grnn_rv_out_size);
        pt_bi_grnn_begin_rv_seq2batch_out_l3 = (float*)
            ((char*)pt_bi_grnn_begin_fw_seq2batch_out_l3
            + pt_bi_grnn_begin_fw_seq2batch_out_size);
    }
    // stage-1 finish
    xpu_wait(ctx_1->xpu_stream);
    // stage-2
    stream_used_l3_size = q_bi_grnn_cdnn.infer(ctx_2, q_id,
        q_bi_grnn_begin.fw_seq2batch.out, q_bi_grnn_begin.rv_seq2batch.out,
        l3_buffer, l3_size,
        q_bi_grnn_cdnn_bi_grnn_fw_out_l3,
        q_bi_grnn_cdnn_bi_grnn_rv_out_l3);
    pt_bi_grnn_begin.infer(ctx_1, pt_id,
        l3_buffer + stream_used_l3_size, l3_size - stream_used_l3_size,
        nullptr, // bid_emb_ew_out_fw_l3
        pt_bi_grnn_begin_fw_seq2batch_out_l3,
        pt_bi_grnn_begin_rv_seq2batch_out_l3);
    l3_manager.free_bottom(); // free stage-1 l3_malloc
    // stage-3 prepare
    // l3_malloc bottom
    // pt_bi_grnn_cdnn.bi_grnn.fw_out
    // pt_bi_grnn_cdnn.bi_grnn.rv_out
    int pt_bi_grnn_cdnn_bi_grnn_fw_out_size =
        pt_bi_grnn_cdnn.bi_grnn.fw_out_size(pt_id);
    int pt_bi_grnn_cdnn_bi_grnn_rv_out_size =
        pt_bi_grnn_cdnn.bi_grnn.rv_out_size(pt_id);
    workspace_need_size = pt_bi_grnn_cdnn_bi_grnn_fw_out_size
        + pt_bi_grnn_cdnn_bi_grnn_rv_out_size;
    workspace_base_addr = l3_manager.malloc_bottom(workspace_need_size);
    l3_buffer = l3_manager.unused_workspace();
    l3_size = l3_manager.unused_size();
    float* pt_bi_grnn_cdnn_bi_grnn_fw_out_l3 = nullptr;
    float* pt_bi_grnn_cdnn_bi_grnn_rv_out_l3 = nullptr;
    if (workspace_base_addr != nullptr) {
        pt_bi_grnn_cdnn_bi_grnn_fw_out_l3 = workspace_base_addr;
        pt_bi_grnn_cdnn_bi_grnn_rv_out_l3 = (float*)
            ((char*)pt_bi_grnn_cdnn_bi_grnn_fw_out_l3
            + pt_bi_grnn_cdnn_bi_grnn_fw_out_size);
    }
    // stage-2 finish
    xpu_wait(ctx_1->xpu_stream);
    xpu_wait(ctx_2->xpu_stream);
    // stage-3
    stream_used_l3_size = pt_bi_grnn_cdnn.infer(ctx_2, pt_id,
        pt_bi_grnn_begin.fw_seq2batch.out, pt_bi_grnn_begin.rv_seq2batch.out,
        l3_buffer, l3_size,
        pt_bi_grnn_cdnn_bi_grnn_fw_out_l3,
        pt_bi_grnn_cdnn_bi_grnn_rv_out_l3);
    q_bi_grnn_end.infer(ctx_1, q_id,
        q_bi_grnn_begin.bid_emb_ew.out_fw,
        q_bi_grnn_cdnn.bi_grnn.fw_out,
        q_bi_grnn_cdnn.bi_grnn.rv_out,
        l3_buffer + stream_used_l3_size, l3_size - stream_used_l3_size);
    pa_emb_ew.infer(ctx_1, pa_id);
    l3_manager.free_top(); // free stage-2 l3_malloc
    // stage-4 prepare
    // l3_malloc top
    // q_pa_match_conv_cdnn.match.out
    // q_pa_match_conv_cdnn.conv.out
    // q_pt_match_conv_cdnn.match.out
    // q_pt_match_conv_cdnn.conv.out
    // pt_bi_grnn_end.concat_2in.out
    int q_pa_match_conv_cdnn_match_out_size =
        q_pa_match_conv_cdnn.match.out_size(q_id, pa_id);
    int q_pa_match_conv_cdnn_conv_out_size =
        q_pa_match_conv_cdnn.conv.out_size(q_id, pa_id);
    int q_pt_match_conv_cdnn_match_out_size =
        q_pt_match_conv_cdnn.match.out_size(q_id, pt_id);
    int q_pt_match_conv_cdnn_conv_out_size =
        q_pt_match_conv_cdnn.conv.out_size(q_id, pt_id);
    int pt_bi_grnn_end_concat_2in_out_size =
        pt_bi_grnn_end.concat_2in.out_size(pt_id.seqlen_sum);
    workspace_need_size = q_pa_match_conv_cdnn_match_out_size
        + q_pa_match_conv_cdnn_conv_out_size
        + q_pt_match_conv_cdnn_match_out_size
        + q_pt_match_conv_cdnn_conv_out_size
        + pt_bi_grnn_end_concat_2in_out_size;
    workspace_base_addr = l3_manager.malloc_top(workspace_need_size);
    l3_buffer = l3_manager.unused_workspace();
    l3_size = l3_manager.unused_size();
    float* q_pa_match_conv_cdnn_match_out_l3 = nullptr;
    float* q_pa_match_conv_cdnn_conv_out_l3 = nullptr;
    float* q_pt_match_conv_cdnn_match_out_l3 = nullptr;
    float* q_pt_match_conv_cdnn_conv_out_l3 = nullptr;
    float* pt_bi_grnn_end_concat_2in_out_l3 = nullptr;
    if (workspace_base_addr != nullptr) {
        q_pa_match_conv_cdnn_match_out_l3 = workspace_base_addr;
        q_pa_match_conv_cdnn_conv_out_l3 = (float*)
            ((char*)q_pa_match_conv_cdnn_match_out_l3
            + q_pa_match_conv_cdnn_match_out_size);
        q_pt_match_conv_cdnn_match_out_l3 = (float*)
            ((char*)q_pa_match_conv_cdnn_conv_out_l3
            + q_pa_match_conv_cdnn_conv_out_size);
        q_pt_match_conv_cdnn_conv_out_l3 = (float*)
            ((char*)q_pt_match_conv_cdnn_match_out_l3
            + q_pt_match_conv_cdnn_match_out_size);
        pt_bi_grnn_end_concat_2in_out_l3 = (float*)
            ((char*)q_pt_match_conv_cdnn_conv_out_l3
            + q_pt_match_conv_cdnn_conv_out_size);
    }
    // stage-3 finish
    xpu_wait(ctx_1->xpu_stream);
    xpu_wait(ctx_2->xpu_stream);
    // stage-4
    stream_used_l3_size = q_pa_match_conv_cdnn.infer(ctx_2, q_id, pa_id,
        q_bi_grnn_begin.bid_emb_ew.out_fw, pa_emb_ew.out,
        l3_buffer, l3_size,
        q_pa_match_conv_cdnn_match_out_l3,
        q_pa_match_conv_cdnn_conv_out_l3);
    pt_bi_grnn_end.infer(ctx_1, pt_id,
        pt_bi_grnn_begin.bid_emb_ew.out_fw,
        pt_bi_grnn_cdnn.bi_grnn.fw_out,
        pt_bi_grnn_cdnn.bi_grnn.rv_out,
        l3_buffer + stream_used_l3_size, l3_size - stream_used_l3_size,
        nullptr, // fw_pool_out_l3
        nullptr, // rv_pool_out_l3
        pt_bi_grnn_end_concat_2in_out_l3,
        nullptr/* concat_3in_out_l3 */);
    q_pt_match_conv_cdnn.infer(ctx_1, q_id, pt_id,
        q_bi_grnn_end.concat_3in.out, pt_bi_grnn_end.concat_3in.out,
        l3_buffer + stream_used_l3_size, l3_size - stream_used_l3_size,
        q_pt_match_conv_cdnn_match_out_l3,
        q_pt_match_conv_cdnn_conv_out_l3);
    l3_manager.free_bottom(); // free stage-3 l3_malloc
    // stage-5 prepare
    // malloc_bottom
    // q_att.out
    // pt_att.out
    // pa_att.out
    // merge_all_begin.fw_seq2batch.out
    // merge_all_begin.rv_seq2batch.out
    int q_att_out_size = q_att.out_size(q_id);
    int pt_att_out_size = pt_att.out_size(pt_id);
    int pa_att_out_size = pa_att.out_size(pa_id);
    int merge_all_begin_fw_seq2batch_out_size =
        merge_all_begin.fw_seq2batch.out_size(q_id);
    int merge_all_begin_rv_seq2batch_out_size =
        merge_all_begin.rv_seq2batch.out_size(q_id);
    workspace_need_size = q_att_out_size
        + pt_att_out_size
        + pa_att_out_size
        + merge_all_begin_fw_seq2batch_out_size
        + merge_all_begin_rv_seq2batch_out_size;
    workspace_base_addr = l3_manager.malloc_bottom(workspace_need_size);
    l3_buffer = l3_manager.unused_workspace();
    l3_size = l3_manager.unused_size();
    float* q_att_out_l3 = nullptr;
    float* pt_att_out_l3 = nullptr;
    float* pa_att_out_l3 = nullptr;
    float* merge_all_begin_fw_seq2batch_out_l3 = nullptr;
    float* merge_all_begin_rv_seq2batch_out_l3 = nullptr;
    if (workspace_base_addr != nullptr) {
        q_att_out_l3 = workspace_base_addr;
        pt_att_out_l3 = (float*)((char*)q_att_out_l3 + q_att_out_size);
        pa_att_out_l3 = (float*)((char*)pt_att_out_l3 + pt_att_out_size);
        merge_all_begin_fw_seq2batch_out_l3 = (float*)
            ((char*)pa_att_out_l3 + pa_att_out_size);
        merge_all_begin_rv_seq2batch_out_l3 = (float*)
            ((char*)merge_all_begin_fw_seq2batch_out_l3
            + merge_all_begin_fw_seq2batch_out_size);
    }
    // stage-4 finish
    xpu_wait(ctx_1->xpu_stream);
    xpu_wait(ctx_2->xpu_stream);
    // stage-5
    int used_l3_size_q_att = q_att.infer(ctx_2, q_id,
        q_bi_grnn_end.concat_2in.out,
        l3_buffer, l3_size,
        q_att_out_l3);
    int used_l3_size_pt_att = pt_att.infer(ctx_2, pt_id,
        pt_bi_grnn_end.concat_2in.out,
        l3_buffer, l3_size,
        pt_att_out_l3);
    int used_l3_size_pa_att = pa_att.infer(ctx_2, pa_id,
        pa_emb_ew.out,
        l3_buffer, l3_size,
        pa_att_out_l3);
    stream_used_l3_size = std::max(used_l3_size_q_att, used_l3_size_pt_att);
    stream_used_l3_size = std::max(stream_used_l3_size, used_l3_size_pa_att);
    merge_all_begin.infer(ctx_1, q_id,
        pa_id,
        q_pa_match_conv_cdnn.match.out, q_pa_match_conv_cdnn.conv.out,
        pt_id,
        q_pt_match_conv_cdnn.match.out, q_pt_match_conv_cdnn.conv.out,
        l3_buffer + stream_used_l3_size, l3_size - stream_used_l3_size,
        merge_all_begin_fw_seq2batch_out_l3,
        merge_all_begin_rv_seq2batch_out_l3);
    l3_manager.free_top(); // free stage-4 l3_malloc
    // stage-5 finish
    xpu_wait(ctx_1->xpu_stream);
    xpu_wait(ctx_2->xpu_stream);
    // stage-6
    merge_all_end.infer(ctx_1, q_id,
        merge_all_begin.fw_seq2batch.out, merge_all_begin.rv_seq2batch.out,
        q_bi_grnn_end.fw_seq_last_pooling.out,
        q_bi_grnn_end.rv_seq_last_pooling.out,
        q_att.out,
        pt_bi_grnn_end.fw_seq_last_pooling.out,
        pt_bi_grnn_end.rv_seq_last_pooling.out,
        pt_att.out,
        pa_att.out,
        l3_buffer, l3_size,
        param.merge_all_out->mutable_data<float>(TARGET(kXPU)));
    l3_manager.free_bottom();// free stage-5 l3_malloc
    xpu_wait(ctx_1->xpu_stream);
}

class XPUMmdnnMultiStreamV2Compute
        : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
public:
    using param_t = operators::XPUMmdnnMultiStreamV2Param;

    void PrepareForRun() override;

    void Run() override;

private:
    int UB_batch = 20;      //  upper bound of batch
    int UB_seqlen = 512;    //  upper bound of seqlen
    api::Context* ctx_1{nullptr};
    api::Context* ctx_2{nullptr};
    IdInfo q_id;
    IdInfo pt_id;
    IdInfo pa_id;
    IdInfo pcq_id;
    BiGrnnBegin q_bi_grnn_begin;
    BiGrnnCDNN q_bi_grnn_cdnn;
    BiGrnnEnd q_bi_grnn_end;
    BiGrnnBegin pt_bi_grnn_begin;
    BiGrnnCDNN pt_bi_grnn_cdnn;
    BiGrnnEnd pt_bi_grnn_end;
    QPcqEmb q_pcq_emb;
    MatchConvCDNN q_pcq_match_conv_cdnn;
    EmbEw pa_emb_ew;
    MatchConvCDNN q_pa_match_conv_cdnn;
    MatchConvCDNN q_pt_match_conv_cdnn;
    Attention q_att;
    Attention pt_att;
    Attention pa_att;
    MergeAllBeginV2 merge_all_begin;
    MergeAllEnd merge_all_end;
};

void XPUMmdnnMultiStreamV2Compute::PrepareForRun() {
    auto& param = this->Param<param_t>();

    XPUStream stream_1;
    int ret = xpu_stream_create(&stream_1);
    if (ret) {
        std::cout << "stream_1 create fail!" << std::endl;
        return;
    }
    ctx_1 = api::create_context();
    api::set_workspace_l3_size(ctx_1, 0);
    ctx_1->xpu_stream = stream_1;

    XPUStream stream_2;
    ret = xpu_stream_create(&stream_2);
    if (ret) {
        std::cout << "stream_2 create fail!" << std::endl;
        return;
    }
    ctx_2 = api::create_context();
    api::set_workspace_l3_size(ctx_2, 0);
    ctx_2->xpu_stream = stream_2;

    const int emb_dim = 128;
    int _cap_e = emb_dim;
    int _cap_h = 128;
    int q_pcq_dim_t = 3;
    int q_pcq_out_channel = 15;
    std::vector<int> q_pcq_topks = {1, 2, 5, 12};
    int q_pa_dim_t = 3;
    int q_pa_out_channel = 5;
    std::vector<int> q_pa_topks = {1, 2, 5, 12};
    int q_pt_dim_t = 5;
    int q_pt_out_channel = 5;
    std::vector<int> q_pt_topks = {1, 2, 4};

    q_id.init(UB_batch, UB_seqlen);
    pt_id.init(UB_batch, UB_seqlen);
    pa_id.init(UB_batch, UB_seqlen);
    pcq_id.init(UB_batch, UB_seqlen);

    q_bi_grnn_begin.init(param.emb_tbl, UB_batch, UB_seqlen);
    q_bi_grnn_cdnn.init(
        param.q_bid_emb_grnn_att_grnn_fw_wh,
        param.q_bid_emb_grnn_att_grnn_fw_wh_maxs,
        param.q_bid_emb_grnn_att_grnn_fw_wi,
        param.q_bid_emb_grnn_att_grnn_fw_wi_maxs,
        param.q_bid_emb_grnn_att_grnn_rv_wh,
        param.q_bid_emb_grnn_att_grnn_rv_wh_maxs,
        param.q_bid_emb_grnn_att_grnn_rv_wi,
        param.q_bid_emb_grnn_att_grnn_rv_wi_maxs,
        _cap_e, _cap_h, UB_batch, UB_seqlen);
    q_bi_grnn_end.init(_cap_e, _cap_h, UB_batch, UB_seqlen);

    pt_bi_grnn_begin.init(param.emb_tbl, UB_batch, UB_seqlen);
    pt_bi_grnn_cdnn.init(
        param.pt_bid_emb_grnn_att_grnn_fw_wh,
        param.pt_bid_emb_grnn_att_grnn_fw_wh_maxs,
        param.pt_bid_emb_grnn_att_grnn_fw_wi,
        param.pt_bid_emb_grnn_att_grnn_fw_wi_maxs,
        param.pt_bid_emb_grnn_att_grnn_rv_wh,
        param.pt_bid_emb_grnn_att_grnn_rv_wh_maxs,
        param.pt_bid_emb_grnn_att_grnn_rv_wi,
        param.pt_bid_emb_grnn_att_grnn_rv_wi_maxs,
        _cap_e, _cap_h, UB_batch, UB_seqlen);
    pt_bi_grnn_end.init(_cap_e, _cap_h, UB_batch, UB_seqlen);

    q_pcq_emb.init(param.emb_tbl, UB_batch, UB_seqlen);
    q_pcq_match_conv_cdnn.init(
        param.q_pcq_match_conv_topk_input_w,
        param.q_pcq_match_conv_topk_input_w_max,
        param.q_pcq_match_conv_topk_conv_w,
        param.q_pcq_match_conv_topk_conv_w_max,
        q_pcq_dim_t, emb_dim, q_pcq_out_channel,
        UB_batch, UB_seqlen);

    pa_emb_ew.init(param.emb_tbl, UB_batch, UB_seqlen);
    q_pa_match_conv_cdnn.init(
        param.q_pa_match_conv_topk_input_w,
        param.q_pa_match_conv_topk_input_w_max,
        param.q_pa_match_conv_topk_conv_w,
        param.q_pa_match_conv_topk_conv_w_max,
        q_pa_dim_t, emb_dim, q_pa_out_channel,
        UB_batch, UB_seqlen);
    q_pt_match_conv_cdnn.init(
        param.q_pt_match_conv_topk_input_w,
        param.q_pt_match_conv_topk_input_w_max,
        param.q_pt_match_conv_topk_conv_w,
        param.q_pt_match_conv_topk_conv_w_max,
        q_pt_dim_t, _cap_h * 3, q_pt_out_channel,
        UB_batch, UB_seqlen);

    q_att.init(
        param.q_bid_emb_grnn_att_att_fc_w,
        param.q_bid_emb_grnn_att_att_fc_w_max,
        param.q_bid_emb_grnn_att_att_fc_b,
        2 * _cap_h, UB_batch, UB_seqlen);
    pt_att.init(
        param.pt_bid_emb_grnn_att_att_fc_w,
        param.pt_bid_emb_grnn_att_att_fc_w_max,
        param.pt_bid_emb_grnn_att_att_fc_b,
        2 * _cap_h, UB_batch, UB_seqlen);
    pa_att.init(
        param.pa_bid_emb_att_att_fc_w,
        param.pa_bid_emb_att_att_fc_w_max,
        param.pa_bid_emb_att_att_fc_b,
        emb_dim, UB_batch, UB_seqlen);

    merge_all_begin.init(
        q_pcq_dim_t, q_pcq_out_channel, q_pcq_topks,
        q_pa_dim_t, q_pa_out_channel, q_pa_topks,
        q_pt_dim_t, q_pt_out_channel, q_pt_topks,
        UB_batch, UB_seqlen);

    merge_all_end.init(
        param.merge_all_grnn_fw_wh,
        param.merge_all_grnn_fw_wh_maxs,
        param.merge_all_grnn_fw_wi,
        param.merge_all_grnn_fw_wi_maxs,
        param.merge_all_grnn_rv_wh,
        param.merge_all_grnn_rv_wh_maxs,
        param.merge_all_grnn_rv_wi,
        param.merge_all_grnn_rv_wi_maxs,
        (30 + 32 + 72), 64,
        param.merge_all_fc0_w,
        param.merge_all_fc0_w_max,
        param.merge_all_fc0_b,
        param.merge_all_fc1_w,
        param.merge_all_fc1_w_max,
        param.merge_all_fc1_b,
        param.merge_all_fc2_w,
        param.merge_all_fc2_w_max,
        param.merge_all_fc2_b,
        UB_batch, UB_seqlen);

    xpu_disp_set_single_mode();
}

void XPUMmdnnMultiStreamV2Compute::Run() {
    auto& param = this->Param<param_t>();
    int batch = param.q_basic->lod()[0].size() - 1;
    if (batch > UB_batch) {
        LOG(FATAL) << "Batch of MMDNN should not be larger than " << UB_batch << std::endl;
    }

    auto& ctx = this->ctx_->As<XPUContext>();
    auto* xpu_ctx = ctx.GetRawContext();
    char* l3_buffer_base = (char*)(xpu_ctx->workspace_l3_ptr);
    int total_l3_size = xpu_ctx->workspace_l3_size;
    q_id.get_xpu_buffer_l3(l3_buffer_base, total_l3_size);
    pt_id.get_xpu_buffer_l3(l3_buffer_base, total_l3_size);
    pa_id.get_xpu_buffer_l3(l3_buffer_base, total_l3_size);
    pcq_id.get_xpu_buffer_l3(l3_buffer_base, total_l3_size);
    L3BottomTopManager l3_manager(l3_buffer_base, total_l3_size);

    int workspace_need_size = 0;
    float* workspace_base_addr = nullptr;
    char* l3_buffer = nullptr;
    int l3_size = 0;
    int stream_used_l3_size = 0;

    // stage-0
    q_id.update(param.q_basic, param.q_bigram0);
    pt_id.update(param.pt_basic, param.pt_bigram0);
    pa_id.update(param.pa_basic, param.pa_bigram0);
    pcq_id.update(param.pcq_basic, param.pcq_basic);
    // stage-1 prepare
    // l3_malloc bottom
    // q_bi_grnn_begin.fw_seq2batch.out
    // q_bi_grnn_begin.rv_seq2batch.out
    int q_bi_grnn_begin_fw_seq2batch_out_size =
        q_bi_grnn_begin.fw_seq2batch.out_size(q_id);
    int q_bi_grnn_begin_rv_seq2batch_out_size =
        q_bi_grnn_begin.rv_seq2batch.out_size(q_id);
    workspace_need_size = q_bi_grnn_begin_fw_seq2batch_out_size
        + q_bi_grnn_begin_rv_seq2batch_out_size;
    workspace_base_addr = l3_manager.malloc_bottom(workspace_need_size);
    l3_buffer = l3_manager.unused_workspace();
    l3_size = l3_manager.unused_size();
    float* q_bi_grnn_begin_fw_seq2batch_out_l3 = nullptr;
    float* q_bi_grnn_begin_rv_seq2batch_out_l3 = nullptr;
    if (workspace_base_addr != nullptr) {
        q_bi_grnn_begin_fw_seq2batch_out_l3 = workspace_base_addr;
        q_bi_grnn_begin_rv_seq2batch_out_l3 = (float*)
            ((char*)q_bi_grnn_begin_fw_seq2batch_out_l3
            + q_bi_grnn_begin_fw_seq2batch_out_size);
    }
    // stage-1
    // ctx_1 only
    q_bi_grnn_begin.infer(ctx_1, q_id,
        l3_buffer, l3_size,
        nullptr, // bid_emb_ew_out_fw_l3
        q_bi_grnn_begin_fw_seq2batch_out_l3,
        q_bi_grnn_begin_rv_seq2batch_out_l3);
    // stage-2 prepare
    // l3_malloc top
    // q_bi_grnn_cdnn.bi_grnn.fw_out
    // q_bi_grnn_cdnn.bi_grnn.rv_out
    // pt_bi_grnn_begin.fw_seq2batch.out
    // pt_bi_grnn_begin.rv_seq2batch.out
    int q_bi_grnn_cdnn_bi_grnn_fw_out_size =
        q_bi_grnn_cdnn.bi_grnn.fw_out_size(q_id);
    int q_bi_grnn_cdnn_bi_grnn_rv_out_size =
        q_bi_grnn_cdnn.bi_grnn.rv_out_size(q_id);
    int pt_bi_grnn_begin_fw_seq2batch_out_size =
        pt_bi_grnn_begin.fw_seq2batch.out_size(pt_id);
    int pt_bi_grnn_begin_rv_seq2batch_out_size =
        pt_bi_grnn_begin.rv_seq2batch.out_size(pt_id);
    workspace_need_size = q_bi_grnn_cdnn_bi_grnn_fw_out_size
        + q_bi_grnn_cdnn_bi_grnn_rv_out_size
        + pt_bi_grnn_begin_fw_seq2batch_out_size
        + pt_bi_grnn_begin_rv_seq2batch_out_size;
    workspace_base_addr = l3_manager.malloc_top(workspace_need_size);
    l3_buffer = l3_manager.unused_workspace();
    l3_size = l3_manager.unused_size();
    float* q_bi_grnn_cdnn_bi_grnn_fw_out_l3 = nullptr;
    float* q_bi_grnn_cdnn_bi_grnn_rv_out_l3 = nullptr;
    float* pt_bi_grnn_begin_fw_seq2batch_out_l3 = nullptr;
    float* pt_bi_grnn_begin_rv_seq2batch_out_l3 = nullptr;
    if (workspace_base_addr != nullptr) {
        q_bi_grnn_cdnn_bi_grnn_fw_out_l3 = workspace_base_addr;
        q_bi_grnn_cdnn_bi_grnn_rv_out_l3 = (float*)
            ((char*)q_bi_grnn_cdnn_bi_grnn_fw_out_l3
            + q_bi_grnn_cdnn_bi_grnn_fw_out_size);
        pt_bi_grnn_begin_fw_seq2batch_out_l3 = (float*)
            ((char*)q_bi_grnn_cdnn_bi_grnn_rv_out_l3
            + q_bi_grnn_cdnn_bi_grnn_rv_out_size);
        pt_bi_grnn_begin_rv_seq2batch_out_l3 = (float*)
            ((char*)pt_bi_grnn_begin_fw_seq2batch_out_l3
            + pt_bi_grnn_begin_fw_seq2batch_out_size);
    }
    // stage-1 finish
    xpu_wait(ctx_1->xpu_stream);
    // stage-2
    stream_used_l3_size = q_bi_grnn_cdnn.infer(ctx_2, q_id,
        q_bi_grnn_begin.fw_seq2batch.out, q_bi_grnn_begin.rv_seq2batch.out,
        l3_buffer, l3_size,
        q_bi_grnn_cdnn_bi_grnn_fw_out_l3,
        q_bi_grnn_cdnn_bi_grnn_rv_out_l3);
    pt_bi_grnn_begin.infer(ctx_1, pt_id,
        l3_buffer + stream_used_l3_size, l3_size - stream_used_l3_size,
        nullptr, // bid_emb_ew_out_fw_l3
        pt_bi_grnn_begin_fw_seq2batch_out_l3,
        pt_bi_grnn_begin_rv_seq2batch_out_l3);
    l3_manager.free_bottom(); // free stage-1 l3_malloc
    // stage-3 prepare
    // l3_malloc bottom
    // pt_bi_grnn_cdnn.bi_grnn.fw_out
    // pt_bi_grnn_cdnn.bi_grnn.rv_out
    int pt_bi_grnn_cdnn_bi_grnn_fw_out_size =
        pt_bi_grnn_cdnn.bi_grnn.fw_out_size(pt_id);
    int pt_bi_grnn_cdnn_bi_grnn_rv_out_size =
        pt_bi_grnn_cdnn.bi_grnn.rv_out_size(pt_id);
    workspace_need_size = pt_bi_grnn_cdnn_bi_grnn_fw_out_size
        + pt_bi_grnn_cdnn_bi_grnn_rv_out_size;
    workspace_base_addr = l3_manager.malloc_bottom(workspace_need_size);
    l3_buffer = l3_manager.unused_workspace();
    l3_size = l3_manager.unused_size();
    float* pt_bi_grnn_cdnn_bi_grnn_fw_out_l3 = nullptr;
    float* pt_bi_grnn_cdnn_bi_grnn_rv_out_l3 = nullptr;
    if (workspace_base_addr != nullptr) {
        pt_bi_grnn_cdnn_bi_grnn_fw_out_l3 = workspace_base_addr;
        pt_bi_grnn_cdnn_bi_grnn_rv_out_l3 = (float*)
            ((char*)pt_bi_grnn_cdnn_bi_grnn_fw_out_l3
            + pt_bi_grnn_cdnn_bi_grnn_fw_out_size);
    }
    // stage-2 finish
    xpu_wait(ctx_1->xpu_stream);
    xpu_wait(ctx_2->xpu_stream);
    // stage-3
    stream_used_l3_size = pt_bi_grnn_cdnn.infer(ctx_2, pt_id,
        pt_bi_grnn_begin.fw_seq2batch.out, pt_bi_grnn_begin.rv_seq2batch.out,
        l3_buffer, l3_size,
        pt_bi_grnn_cdnn_bi_grnn_fw_out_l3,
        pt_bi_grnn_cdnn_bi_grnn_rv_out_l3);
    q_bi_grnn_end.infer(ctx_1, q_id,
        q_bi_grnn_begin.bid_emb_ew.out_fw,
        q_bi_grnn_cdnn.bi_grnn.fw_out,
        q_bi_grnn_cdnn.bi_grnn.rv_out,
        l3_buffer + stream_used_l3_size, l3_size - stream_used_l3_size);
    pa_emb_ew.infer(ctx_1, pa_id);
    q_pcq_emb.infer(ctx_1, q_id, pcq_id);
    l3_manager.free_top(); // free stage-2 l3_malloc
    // stage-4 prepare
    // l3_malloc top
    // q_pcq_match_conv_cdnn.match.out
    // q_pcq_match_conv_cdnn.conv.out
    // q_pa_match_conv_cdnn.match.out
    // q_pa_match_conv_cdnn.conv.out
    // q_pt_match_conv_cdnn.match.out
    // q_pt_match_conv_cdnn.conv.out
    // pt_bi_grnn_end.concat_2in.out
    int q_pcq_match_conv_cdnn_match_out_size =
        q_pcq_match_conv_cdnn.match.out_size(q_id, pcq_id);
    int q_pcq_match_conv_cdnn_conv_out_size =
        q_pcq_match_conv_cdnn.conv.out_size(q_id, pcq_id);
    int q_pa_match_conv_cdnn_match_out_size =
        q_pa_match_conv_cdnn.match.out_size(q_id, pa_id);
    int q_pa_match_conv_cdnn_conv_out_size =
        q_pa_match_conv_cdnn.conv.out_size(q_id, pa_id);
    int q_pt_match_conv_cdnn_match_out_size =
        q_pt_match_conv_cdnn.match.out_size(q_id, pt_id);
    int q_pt_match_conv_cdnn_conv_out_size =
        q_pt_match_conv_cdnn.conv.out_size(q_id, pt_id);
    int pt_bi_grnn_end_concat_2in_out_size =
        pt_bi_grnn_end.concat_2in.out_size(pt_id.seqlen_sum);
    workspace_need_size = q_pcq_match_conv_cdnn_match_out_size
        + q_pcq_match_conv_cdnn_conv_out_size
        + q_pa_match_conv_cdnn_match_out_size
        + q_pa_match_conv_cdnn_conv_out_size
        + q_pt_match_conv_cdnn_match_out_size
        + q_pt_match_conv_cdnn_conv_out_size
        + pt_bi_grnn_end_concat_2in_out_size;
    workspace_base_addr = l3_manager.malloc_top(workspace_need_size);
    l3_buffer = l3_manager.unused_workspace();
    l3_size = l3_manager.unused_size();
    float* q_pcq_match_conv_cdnn_match_out_l3 = nullptr;
    float* q_pcq_match_conv_cdnn_conv_out_l3 = nullptr;
    float* q_pa_match_conv_cdnn_match_out_l3 = nullptr;
    float* q_pa_match_conv_cdnn_conv_out_l3 = nullptr;
    float* q_pt_match_conv_cdnn_match_out_l3 = nullptr;
    float* q_pt_match_conv_cdnn_conv_out_l3 = nullptr;
    float* pt_bi_grnn_end_concat_2in_out_l3 = nullptr;
    if (workspace_base_addr != nullptr) {
        q_pcq_match_conv_cdnn_match_out_l3 = workspace_base_addr;
        q_pcq_match_conv_cdnn_conv_out_l3 = (float*)
            ((char*)q_pcq_match_conv_cdnn_match_out_l3
            + q_pcq_match_conv_cdnn_match_out_size);
        q_pa_match_conv_cdnn_match_out_l3 = (float*)
            ((char*)q_pcq_match_conv_cdnn_conv_out_l3
            + q_pcq_match_conv_cdnn_conv_out_size);
        q_pa_match_conv_cdnn_conv_out_l3 = (float*)
            ((char*)q_pa_match_conv_cdnn_match_out_l3
            + q_pa_match_conv_cdnn_match_out_size);
        q_pt_match_conv_cdnn_match_out_l3 = (float*)
            ((char*)q_pa_match_conv_cdnn_conv_out_l3
            + q_pa_match_conv_cdnn_conv_out_size);
        q_pt_match_conv_cdnn_conv_out_l3 = (float*)
            ((char*)q_pt_match_conv_cdnn_match_out_l3
            + q_pt_match_conv_cdnn_match_out_size);
        pt_bi_grnn_end_concat_2in_out_l3 = (float*)
            ((char*)q_pt_match_conv_cdnn_conv_out_l3
            + q_pt_match_conv_cdnn_conv_out_size);
    }
    // stage-3 finish
    xpu_wait(ctx_1->xpu_stream);
    xpu_wait(ctx_2->xpu_stream);
    // stage-4
    int used_l3_size_q_pcq = q_pcq_match_conv_cdnn.infer(ctx_2, q_id, pcq_id,
        q_pcq_emb.out_q, q_pcq_emb.out_pcq,
        l3_buffer, l3_size,
        q_pcq_match_conv_cdnn_match_out_l3,
        q_pcq_match_conv_cdnn_conv_out_l3);
    int used_l3_size_q_pa = q_pa_match_conv_cdnn.infer(ctx_2, q_id, pa_id,
        q_bi_grnn_begin.bid_emb_ew.out_fw, pa_emb_ew.out,
        l3_buffer, l3_size,
        q_pa_match_conv_cdnn_match_out_l3,
        q_pa_match_conv_cdnn_conv_out_l3);
    stream_used_l3_size = std::max(used_l3_size_q_pcq, used_l3_size_q_pa);
    pt_bi_grnn_end.infer(ctx_1, pt_id,
        pt_bi_grnn_begin.bid_emb_ew.out_fw,
        pt_bi_grnn_cdnn.bi_grnn.fw_out,
        pt_bi_grnn_cdnn.bi_grnn.rv_out,
        l3_buffer + stream_used_l3_size, l3_size - stream_used_l3_size,
        nullptr, // fw_pool_out_l3
        nullptr, // rv_pool_out_l3
        pt_bi_grnn_end_concat_2in_out_l3,
        nullptr/* concat_3in_out_l3 */);
    q_pt_match_conv_cdnn.infer(ctx_1, q_id, pt_id,
        q_bi_grnn_end.concat_3in.out, pt_bi_grnn_end.concat_3in.out,
        l3_buffer + stream_used_l3_size, l3_size - stream_used_l3_size,
        q_pt_match_conv_cdnn_match_out_l3,
        q_pt_match_conv_cdnn_conv_out_l3);
    l3_manager.free_bottom(); // free stage-3 l3_malloc
    // stage-5 prepare
    // malloc_bottom
    // q_att.out
    // pt_att.out
    // pa_att.out
    // merge_all_begin.fw_seq2batch.out
    // merge_all_begin.rv_seq2batch.out
    int q_att_out_size = q_att.out_size(q_id);
    int pt_att_out_size = pt_att.out_size(pt_id);
    int pa_att_out_size = pa_att.out_size(pa_id);
    int merge_all_begin_fw_seq2batch_out_size =
        merge_all_begin.fw_seq2batch.out_size(q_id);
    int merge_all_begin_rv_seq2batch_out_size =
        merge_all_begin.rv_seq2batch.out_size(q_id);
    workspace_need_size = q_att_out_size
        + pt_att_out_size
        + pa_att_out_size
        + merge_all_begin_fw_seq2batch_out_size
        + merge_all_begin_rv_seq2batch_out_size;
    workspace_base_addr = l3_manager.malloc_bottom(workspace_need_size);
    l3_buffer = l3_manager.unused_workspace();
    l3_size = l3_manager.unused_size();
    float* q_att_out_l3 = nullptr;
    float* pt_att_out_l3 = nullptr;
    float* pa_att_out_l3 = nullptr;
    float* merge_all_begin_fw_seq2batch_out_l3 = nullptr;
    float* merge_all_begin_rv_seq2batch_out_l3 = nullptr;
    if (workspace_base_addr != nullptr) {
        q_att_out_l3 = workspace_base_addr;
        pt_att_out_l3 = (float*)((char*)q_att_out_l3 + q_att_out_size);
        pa_att_out_l3 = (float*)((char*)pt_att_out_l3 + pt_att_out_size);
        merge_all_begin_fw_seq2batch_out_l3 = (float*)
            ((char*)pa_att_out_l3 + pa_att_out_size);
        merge_all_begin_rv_seq2batch_out_l3 = (float*)
            ((char*)merge_all_begin_fw_seq2batch_out_l3
            + merge_all_begin_fw_seq2batch_out_size);
    }
    // stage-4 finish
    xpu_wait(ctx_1->xpu_stream);
    xpu_wait(ctx_2->xpu_stream);
    // stage-5
    int used_l3_size_q_att = q_att.infer(ctx_2, q_id,
        q_bi_grnn_end.concat_2in.out,
        l3_buffer, l3_size,
        q_att_out_l3);
    int used_l3_size_pt_att = pt_att.infer(ctx_2, pt_id,
        pt_bi_grnn_end.concat_2in.out,
        l3_buffer, l3_size,
        pt_att_out_l3);
    int used_l3_size_pa_att = pa_att.infer(ctx_2, pa_id,
        pa_emb_ew.out,
        l3_buffer, l3_size,
        pa_att_out_l3);
    stream_used_l3_size = std::max(used_l3_size_q_att, used_l3_size_pt_att);
    stream_used_l3_size = std::max(stream_used_l3_size, used_l3_size_pa_att);
    merge_all_begin.infer(ctx_1, q_id,
        pcq_id,
        q_pcq_match_conv_cdnn.match.out, q_pcq_match_conv_cdnn.conv.out,
        pa_id,
        q_pa_match_conv_cdnn.match.out, q_pa_match_conv_cdnn.conv.out,
        pt_id,
        q_pt_match_conv_cdnn.match.out, q_pt_match_conv_cdnn.conv.out,
        l3_buffer + stream_used_l3_size, l3_size - stream_used_l3_size,
        merge_all_begin_fw_seq2batch_out_l3,
        merge_all_begin_rv_seq2batch_out_l3);
    l3_manager.free_top(); // free stage-4 l3_malloc
    // stage-5 finish
    xpu_wait(ctx_1->xpu_stream);
    xpu_wait(ctx_2->xpu_stream);
    // stage-6
    merge_all_end.infer(ctx_1, q_id,
        merge_all_begin.fw_seq2batch.out, merge_all_begin.rv_seq2batch.out,
        q_bi_grnn_end.fw_seq_last_pooling.out,
        q_bi_grnn_end.rv_seq_last_pooling.out,
        q_att.out,
        pt_bi_grnn_end.fw_seq_last_pooling.out,
        pt_bi_grnn_end.rv_seq_last_pooling.out,
        pt_att.out,
        pa_att.out,
        l3_buffer, l3_size,
        param.merge_all_out->mutable_data<float>(TARGET(kXPU)));
    l3_manager.free_bottom();// free stage-5 l3_malloc
    xpu_wait(ctx_1->xpu_stream);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__mmdnn_multi_stream_v1,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUMmdnnMultiStreamV1Compute,
                     def)
    .BindInput("emb_tbl", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_basic", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("q_bigram0", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("pt_basic", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("pt_bigram0", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("pa_basic", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("pa_bigram0", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("q_bid_emb_grnn_att_grnn_fw_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_bid_emb_grnn_att_grnn_fw_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_bid_emb_grnn_att_grnn_rv_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_bid_emb_grnn_att_grnn_rv_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_bid_emb_grnn_att_att_fc_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_bid_emb_grnn_att_att_fc_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pt_bid_emb_grnn_att_grnn_fw_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pt_bid_emb_grnn_att_grnn_fw_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pt_bid_emb_grnn_att_grnn_rv_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pt_bid_emb_grnn_att_grnn_rv_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pt_bid_emb_grnn_att_att_fc_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pt_bid_emb_grnn_att_att_fc_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pa_bid_emb_att_att_fc_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pa_bid_emb_att_att_fc_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_pa_match_conv_topk_input_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_pa_match_conv_topk_conv_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_pt_match_conv_topk_input_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_pt_match_conv_topk_conv_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_grnn_fw_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_grnn_fw_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_grnn_rv_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_grnn_rv_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_fc0_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_fc0_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_fc1_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_fc1_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_fc2_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_fc2_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("merge_all_out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(__xpu__mmdnn_multi_stream_v2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUMmdnnMultiStreamV2Compute,
                     def)
    .BindInput("emb_tbl", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_basic", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("q_bigram0", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("pt_basic", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("pt_bigram0", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("pa_basic", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("pa_bigram0", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("pcq_basic", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("q_bid_emb_grnn_att_grnn_fw_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_bid_emb_grnn_att_grnn_fw_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_bid_emb_grnn_att_grnn_rv_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_bid_emb_grnn_att_grnn_rv_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_bid_emb_grnn_att_att_fc_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_bid_emb_grnn_att_att_fc_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pt_bid_emb_grnn_att_grnn_fw_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pt_bid_emb_grnn_att_grnn_fw_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pt_bid_emb_grnn_att_grnn_rv_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pt_bid_emb_grnn_att_grnn_rv_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pt_bid_emb_grnn_att_att_fc_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pt_bid_emb_grnn_att_att_fc_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pa_bid_emb_att_att_fc_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("pa_bid_emb_att_att_fc_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_pa_match_conv_topk_input_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_pa_match_conv_topk_conv_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_pt_match_conv_topk_input_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_pt_match_conv_topk_conv_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_pcq_match_conv_topk_input_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("q_pcq_match_conv_topk_conv_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_grnn_fw_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_grnn_fw_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_grnn_rv_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_grnn_rv_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_fc0_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_fc0_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_fc1_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_fc1_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_fc2_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("merge_all_fc2_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("merge_all_out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
