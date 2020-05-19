/*!\file gemm_test.cpp
 *
 * \brief DESCRIPTION HERE
 *
 * \author isa@baidu.com
 *
 * \copyright (C) 2018 Baidu, Inc
 */
#include <cstdio>
#include <random>
#include <fstream>
#include <cmath>
#include <iostream>
#include "xpu/api.h"
#include "xpu/runtime.h"
#include <sys/time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#define timeval_to_us(tv) (((tv).tv_sec * 1000000ULL) + (tv).tv_usec)

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "lite/api/lite_api_test_helper.h"
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/utils/cp_logging.h"

namespace api = baidu::xpu::api;
using namespace std;
using namespace paddle;
using namespace paddle::lite;

std::vector<int> get_all_ids(const std::string& dir_in) {
    std::vector<int> ids;
    struct stat s;
    stat(dir_in.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        return ids;
    }
    DIR* open_dir = opendir(dir_in.c_str());
    if (nullptr == open_dir) {
        return ids;
    }
    dirent* p = nullptr;
    while((p = readdir(open_dir)) != nullptr) {
        if (p->d_name[0] != '.') {
            std::string filename = std::string(p->d_name);
            int end_pos = filename.find('_');
            int qid = atoi(filename.substr(0, end_pos).c_str());
            ids.push_back(qid);
        }
    }
    closedir(open_dir);
    return ids;
}

float* get_cpu_data(std::string fname, int size) {
    float* buffer = (float*) malloc(size * sizeof(float));
    ifstream inF;
    inF.open(fname, ifstream::binary);
    if(!inF.read(reinterpret_cast<char*>(buffer), size * sizeof(float))) {
        std::cout << "something wrong" << std::endl;
    }
    inF.close();
    return buffer;
}

long round_even(float in) {
    long ret = llround(in);
    if ((fabs(round(in) - in) >= 0.5f) && (abs(ret) % 2 != 0)) {
        ret += (ret > 0 ? -1 : 1);
    }
    return ret;
}

void float2fix_int16(float* cpu_in, int size, int16_t** xpu_out, float** max_out) {
    float max_cpu[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int16_t* cpu_fix = (int16_t*) malloc(size * sizeof(int16_t));
    xpu_malloc((void**)(max_out), 4 * sizeof(float));
    xpu_malloc((void**)(xpu_out), size * sizeof(int16_t));
    float abs_max = 0.0f;
    for (int i = 0; i < size; i++) {
        if (fabs(cpu_in[i]) > abs_max) {
            abs_max = fabs(cpu_in[i]);
        }
    }
    float quant = 32767 / abs_max;
    for (int i = 0; i < size; i++) {
        cpu_fix[i] = (int16_t)round_even(cpu_in[i] * quant);
    }
    xpu_memcpy((void*)(*xpu_out), (void*)cpu_fix, size * sizeof(int16_t), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    max_cpu[0] = abs_max;
    xpu_memcpy((void*)(*max_out), (void*)max_cpu, 4 * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
}

void float2fix_int31(float* cpu_in, int size, float** xpu_out, float** max_out) {
    float max_cpu[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    xpu_malloc((void**)(max_out), 4 * sizeof(float));
    xpu_malloc((void**)(xpu_out), size * sizeof(float));
    float abs_max = 0.0f;
    for (int i = 0; i < size; i++) {
        if (fabs(cpu_in[i]) > abs_max) {
            abs_max = fabs(cpu_in[i]);
        }
    }
    xpu_memcpy((void*)(*xpu_out), (void*)cpu_in, size * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    max_cpu[0] = abs_max;
    xpu_memcpy((void*)(*max_out), (void*)max_cpu, 4 * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
}
float* get_xpu_data(std::string fname, int size) {
    float* buffer = get_cpu_data(fname, size);
    float* buffer_xpu = nullptr;
    xpu_malloc((void**)(&buffer_xpu), size * sizeof(float));
    xpu_memcpy((void*)buffer_xpu, (void*)buffer, size * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    free(buffer);
    return buffer_xpu;
}

void store_cpu_data(float* buffer, std::string fname, int size) {
    std::ofstream ouF;
    ouF.open(fname, std::ofstream::binary);
    ouF.write(reinterpret_cast<char*>(buffer), sizeof(float) * size);
    ouF.close();
}

void store_xpu_data(float* buffer_xpu, std::string fname, int size) {
    float* buffer = (float*) malloc(size * sizeof(float));
    xpu_memcpy((void*)buffer, (void*)buffer_xpu, size * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    std::ofstream ouF;
    ouF.open(fname, std::ofstream::binary);
    ouF.write(reinterpret_cast<char*>(buffer), sizeof(float) * size);
    ouF.close();
    free(buffer);
}
void get_conv_bn(string conv_name, string bn_name, int f, int c, int h, int w,
        int16_t** out_conv, float** out_bias, float** out_maxptr) {
    float* conv_data = get_cpu_data(conv_name + ".weight", f * c * h * w);
    float* bn_scale = get_cpu_data(bn_name + ".weight", f);
    float* bn_bias = get_cpu_data(bn_name + ".bias", f);
    float* bn_mean = get_cpu_data(bn_name + ".running_mean", f);
    float* bn_var = get_cpu_data(bn_name + ".running_var", f);
    for (int i = 0; i < f; i++) {
        float eps = 1e-5;
        bn_var[i] = sqrt(bn_var[i] + eps);
        bn_scale[i] = bn_scale[i] / bn_var[i];
        bn_bias[i] = bn_bias[i] - bn_mean[i] * bn_scale[i];
    }
    for (int i = 0; i < f * c * h * w; i++) {
        int fid = (i / (c * h * w)) % f;
        conv_data[i] *= bn_scale[fid];
    }
    xpu_malloc((void**)(out_conv), f * c * h * w * sizeof(float));
    xpu_malloc((void**)(out_bias), f * sizeof(float));
    xpu_memcpy((void*)(*out_bias), (void*)bn_bias, f * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    float2fix_int16(conv_data, f * c * h * w, out_conv, out_maxptr);
    free(conv_data);
    free(bn_scale);
    free(bn_bias);
    free(bn_mean);
    free(bn_var);
}

vector<vector<void*>> get_resnet_cbam_conv_info() {
    vector<vector<void*>> result;
    result.resize(4);
    string prefix = "epoch_13_weights/save_weights/";
    int16_t* weight;
    float* bias;
    float* maxptr;
    get_conv_bn(prefix + "features.conv1", prefix + "features.bn1", 64, 3, 7, 7, &weight, &bias, &maxptr);
    result[0].push_back(weight);
    result[1].push_back(bias);
    result[2].push_back(maxptr);
    int block[4] = {3, 4, 6, 3};
    int block_c[8] = {64, 256, 256, 512, 512, 1024, 1024, 2048};
    int block_f[8] = {256, 256, 512, 512, 1024, 1024, 2048, 2048};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < block[i]; j++) {
            string block_prefix = prefix + "features.layer" + to_string(i + 1) + "." + to_string(j) + ".";
            int idx = i * 2 + ((j == 0) ? 0 : 1);
            int in_c[5] = {block_c[idx], block_f[idx] / 4, block_f[idx] / 4, 2, block_c[idx]};
            int out_f[5] = {block_f[idx] / 4, block_f[idx] / 4, block_f[idx], 1, block_f[idx]};
            int windows[5] = {1, 3, 1, 7, 1};
            for (int k = 0; k < 3; k++) {
                string conv_name = block_prefix + "conv" + to_string(k + 1);
                string bn_name = block_prefix + "bn" + to_string(k + 1);
                get_conv_bn(conv_name, bn_name, out_f[k], in_c[k], windows[k], windows[k], &weight, &bias, &maxptr);
                result[0].push_back(weight);
                result[1].push_back(bias);
                result[2].push_back(maxptr);
            }
            // sa-conv
            float* sa_weight = get_cpu_data(block_prefix + "sa.conv1.weight", 2 * 1 * 7 * 7);
            float2fix_int16(sa_weight, 2 * 1 * 7 * 7, &weight, &maxptr);
            result[0].push_back(weight);
            result[1].push_back(nullptr);
            result[2].push_back(maxptr);
            if (j == 0) {
                get_conv_bn(block_prefix + "downsample.0", block_prefix + "downsample.1",
                        out_f[4], in_c[4], windows[4], windows[4], &weight, &bias, &maxptr);
                result[0].push_back(weight);
                result[1].push_back(bias);
                result[2].push_back(maxptr);
            }
        }
    }
    float* fc_weight_cpu = get_cpu_data(prefix + "fc_out.0.weight", 2048 * 64);
    bias = get_xpu_data(prefix + "fc_out.0.bias", 64);
    float2fix_int16(fc_weight_cpu, 2048 * 64, &weight, &maxptr);      // int16
    result[0].push_back(weight);                                      // int16
    // float* fc_weight_int31;                                                 // int31
    // float2fix_int31(fc_weight_cpu, 2048 * 64, &fc_weight_int31, &maxptr);   // int31
    // result[0].push_back(fc_weight_int31);                                   // int31

    result[1].push_back(bias);
    result[2].push_back(maxptr);

    result[3].push_back(get_cpu_data(prefix + "pool.p", 1));
    return result;
}

float get_sum(float* buffer_xpu, int size) {
    float* buffer = (float*) malloc(size * sizeof(float));
    xpu_memcpy((void*)buffer, (void*)buffer_xpu, size * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += buffer[i];
    }
    free(buffer);
    return sum;
}

// ./cbam_test "random" n c h w loop
// ./cbam_test "accuracy" logid
int main(int argc, char** argv) {
    if ((argc != 2) && (argc != 3) && (argc != 7)) {
        std::cout << "cbam_test only support following 2 parameters:" << std::endl;
        std::cout << "\t./cbam_test string(random) n c h w loop" << std::endl;
        std::cout << "\t./cbam_test string(accuracy) [dev_id]" << std::endl;
        exit(1);
    }
    int n = 1;
    int c = 3;
    int height = 224;
    int width = 224;
    int loop = 1;
    (void)loop;
    int dev_id = 0;
    //bool use_fp16 = true;
    std::string mode = std::string(argv[1]);
    std::string out_file = "";
    if (mode == "random") {
        n = atoi(argv[2]);
        c = atoi(argv[3]);
        height = atoi(argv[4]);
        width = atoi(argv[5]);
        loop = atoi(argv[6]);
        if (height > 512 || width > 512) {
            std::cout << "assert fail: height <= 512 && width <= 512" << std::endl;
            exit(1);
        }
    } else if (mode == "accuracy") {
        if (argc == 3) {
            dev_id = atoi(argv[2]);
        }
        // do nothing
    } else {
        std::cout << "invalid mode" << std::endl;
        exit(1);
    }
    //xpu_set_device(dev_id);
    struct timeval start_t;
    struct timeval end_t;
    float tspan_us = 0;

    //auto ctx = api::create_context();
    //api::set_workspace_l3_size(ctx, 16776192);


    //vector<vector<void*>> conv_info = get_resnet_cbam_conv_info();
    //const int16_t** weight_list = (const int16_t**) conv_info[0].data();
    //const float** bias_list = (const float**) conv_info[1].data();
    //const float** max_filter_list = (const float**) conv_info[2].data();
    //float pool_p = ((float**)conv_info[3].data())[0][0];

    //bool dynamic_shape = (mode == "accuracy");

    if (mode == "random") {
#if 0
        int bottom_size = n * 3 * height * width;
        int top_size = n * 64;
        float* bottom_cpu = (float*) malloc(bottom_size * sizeof(float));
        float* top_cpu = (float*) malloc(top_size * sizeof(float));
        float* bottom = nullptr;
        float* top = nullptr;
        xpu_malloc((void**)(&bottom), bottom_size * sizeof(float));
        xpu_malloc((void**)(&top), top_size * sizeof(float));
        char shape_str[100];
        sprintf(shape_str, "[%d, %d, %d, %d]", n, 3, height, width);
        for (int i = 0; i < n * 3 * height * width; i++) {
            float val = (i % 255) / 255.0f;
            bottom_cpu[i] = val;
        }
        // warm-up first
        std::cout << "warm-up: " << shape_str << std::endl;
        xpu_memcpy((void*)bottom, (void*)bottom_cpu, bottom_size * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
        api::conv2d_int16_resnet_cbam<float, int16_t>(ctx, n, height, width, bottom, weight_list, top, bias_list, max_filter_list, pool_p, use_fp16, dynamic_shape);
        xpu_memcpy((void*)top_cpu, (void*)top, top_size * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        xpu_wait();
        // start test
        std::cout << "start" << std::endl;
        gettimeofday(&start_t, NULL);
        for (int i = 0; i < loop; i++) {
            xpu_memcpy((void*)bottom, (void*)bottom_cpu, bottom_size * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
            api::conv2d_int16_resnet_cbam<float, int16_t>(ctx, n, height, width, bottom, weight_list, top, bias_list, max_filter_list, pool_p, use_fp16, dynamic_shape);
            xpu_memcpy((void*)top_cpu, (void*)top, top_size * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        }
        gettimeofday(&end_t, NULL);
        if (loop > 0) {
            tspan_us = (timeval_to_us(end_t) - timeval_to_us(start_t)) / loop;
            std::cout << shape_str << ": " << tspan_us << " us, " << n * 1e6 / tspan_us << " QPS" << std::endl;
        }
#endif
    } else {
  lite_api::CxxConfig config;
  //config.set_model_dir(FLAGS_model_dir);
  std::string model_dir2 = "cbam_package/fingerprint_paddle";
  config.set_model_file(model_dir2 + "/__model__");
  config.set_param_file(model_dir2 + "/__params__");
  config.set_valid_places({lite_api::Place{TARGET(kXPU), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  config.set_xpu_workspace_l3_size_per_thread();
  auto predictor = lite_api::CreatePaddlePredictor(config);


        int total_n = 0;
        std::vector<int> ids = get_all_ids("random_data_0407/logid_file/");
        for (int x = 0; x < ids.size(); x++) {
            int logid = ids[x];
            // get shape info from files
            std::string shape_file = std::string("random_data_0407/logid_file/") + to_string(logid) + "_shape.txt";
            std::string in_file = std::string("random_data_0407/binary_in_data/") + to_string(logid) + "_in.dat";
            out_file = std::string("random_data_0407/xpu_out_data_pd") + to_string(dev_id) + "/" + to_string(logid) + "_out.dat";
            ifstream inF;
            inF.open(shape_file);
            if (!inF) {
                std::cout << "error when reading " << shape_file << std::endl;
                exit(1);
            }
            char useless;
            inF >> useless >> n >> useless >> c >> useless >> height >> useless >> width;
            inF.close();
            char shape_str[100];
            sprintf(shape_str, "[%d, %d, %d, %d]", n, 3, height, width);
            std::cout << "pd" << dev_id << " processing " << shape_str << std::endl;

            int bottom_size = n * 3 * height * width;
            int top_size = n * 64;
            float* bottom_cpu = get_cpu_data(in_file, bottom_size);
            float* top_cpu = (float*) malloc(top_size * sizeof(float));
            //float* bottom = nullptr;
            //float* top = nullptr;
            //xpu_malloc((void**)(&bottom), bottom_size * sizeof(float));
            //xpu_malloc((void**)(&top), top_size * sizeof(float));

            gettimeofday(&start_t, NULL);
            const int infer_maxbatch = 8;
            const int max_async_request_cnt = 4;
            int curr_async_request_cnt = 0;
            for (int i = 0 ; i < n; i += infer_maxbatch) {
                struct timeval loop_start_t;
                struct timeval loop_end_t;
                //float* curr_bottom = bottom + i * 3 * height * width;
                //float* curr_top = top + i * 64;
                float* curr_bottom_cpu = bottom_cpu + i * 3 * height * width;
                float* curr_top_cpu = top_cpu + i * 64;
                int infer_batch = std::min<int>(infer_maxbatch, n - i);
                gettimeofday(&loop_start_t, NULL);
                //xpu_memcpy((void*)curr_bottom, (void*)curr_bottom_cpu, infer_batch * 3 * height * width * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
                //api::conv2d_int16_resnet_cbam<float, int16_t>(ctx, infer_batch, height, width, curr_bottom,
                        //weight_list, curr_top, bias_list, max_filter_list, pool_p, use_fp16, dynamic_shape);

  auto input_tensor = predictor->GetInput(0);
  std::vector<int64_t> input_shape{infer_batch, 3, height, width};
  input_tensor->Resize(input_shape);
  auto* data = input_tensor->mutable_data<float>();
  int input_num = 1;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    input_num *= input_shape[i];
  }
  memcpy(data, curr_bottom_cpu, input_num * sizeof(float));
  //for (int i = 0; i < input_num; i++) {
    //data[i] = 1;
  //}

  predictor->Run();
  auto out = predictor->GetOutput(0);
  memcpy(curr_top_cpu, out->data<float>(), infer_batch * 64 * sizeof(float));

                curr_async_request_cnt++;
                if (curr_async_request_cnt == max_async_request_cnt) {
                    curr_async_request_cnt = 0;
                    xpu_wait();
                    gettimeofday(&loop_end_t, NULL);
                    std::cout << "Requests-Latency: " << (timeval_to_us(loop_end_t) - timeval_to_us(loop_start_t)) / 1000.0 << " ms" << std::endl;
                }
            }
            //xpu_memcpy((void*)top_cpu, (void*)top, n * 64 * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
            gettimeofday(&end_t, NULL);

            //store_xpu_data(top, out_file, n * 64);
            store_cpu_data(top_cpu, out_file, n * 64);

            free(bottom_cpu);
            free(top_cpu);
            //xpu_free(bottom);
            //xpu_free(top);

            tspan_us = tspan_us + (timeval_to_us(end_t) - timeval_to_us(start_t));
            total_n = total_n + n;
        }
        std::cout << "Final-Result: " << tspan_us / 1000 << " ms, " << total_n * 1e6 / tspan_us << " QPS" << std::endl;
    }
    //for (int i = 0; i < 3; i++) {
        //for (int j = 0; j < conv_info[i].size(); j++) {
            //if (conv_info[i][j] != nullptr) {
                //xpu_free(conv_info[i][j]);
            //}
        //}
    //}
    return 0;
}
