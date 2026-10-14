// ==================================================================
//  车道线检测 —— 推理基准测试（OpenVINO 版）
//
//  用同一张图循环跑，分三段计时：预处理 / 推理 / 后处理
//  单图循环的变量最少（帧数固定、没有读帧和解码干扰），适合测纯推理性能
//  参数从 config.json 读，改模型和设备不用重新编译
// ==================================================================

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <chrono>
#include "config.hpp"
using namespace std;

const char* CONFIG_PATH = "config.json";

const int WARMUP = 30;      // 前 30 轮热身，不计入统计
const int LOOPS  = 500;     // 统计 500 轮

struct Detection{
    double cx, cy, w, h;
    int    class_id;
    double score;
};

float iou(const Detection& a, const Detection& b) {
    float ax1 = a.cx - a.w/2, ay1 = a.cy - a.h/2, ax2 = a.cx + a.w/2, ay2 = a.cy + a.h/2;
    float bx1 = b.cx - b.w/2, by1 = b.cy - b.h/2, bx2 = b.cx + b.w/2, by2 = b.cy + b.h/2;
    float iw = min(ax2, bx2) - max(ax1, bx1);
    float ih = min(ay2, by2) - max(ay1, by1);
    if (iw < 0) iw = 0;
    if (ih < 0) ih = 0;
    float inter = iw * ih;
    return inter / (a.w*a.h + b.w*b.h - inter);
}

bool compareScore(const Detection& a, const Detection& b) { return a.score > b.score; }


int main() {
    try {
        Config cfg = load_config(CONFIG_PATH);

        int W_IN  = cfg.input_width;
        int H_IN  = cfg.input_height;
        int N_CLS = (int)cfg.class_names.size();

        // 读一张图，后面一直复用（不进循环，避免把读图算进去）
        cv::Mat img = cv::imread(cfg.image);
        if (img.empty()) { cout << "读图失败：" << cfg.image << endl; return 1; }

        cout << "配置：" << CONFIG_PATH << endl;
        cout << "图片：" << cfg.image << "   " << img.cols << " x " << img.rows << endl;
        cout << "模型：" << cfg.model_ov << endl;
        cout << "设备：" << cfg.device << endl;
        cout << "输入尺寸：" << W_IN << " x " << H_IN << endl;
        cout << "跑 " << (WARMUP + LOOPS) << " 轮（前 " << WARMUP << " 轮热身）...\n" << endl;

        // 加载模型
        ov::Core core;
        auto model    = core.read_model(cfg.model_ov);
        
        auto compiled = core.compile_model(model, cfg.device);
        auto infer    = compiled.create_infer_request();
        ov::Shape input_shape = { 1, 3, (size_t)H_IN, (size_t)W_IN };

        double    t_pre = 0, t_infer = 0, t_post = 0;
        long long box_pre = 0, box_post = 0;

        for (int k = 0; k < WARMUP + LOOPS; k++) {

            // ---------- ① 预处理 ----------
            auto a = chrono::steady_clock::now();
            cv::Mat blob = cv::dnn::blobFromImage(
                img, 1.0/255.0, cv::Size(W_IN, H_IN),
                cv::Scalar(), true, false);
            auto b = chrono::steady_clock::now();

            // ---------- ② 推理 ----------
            ov::Tensor input_tensor(ov::element::f32, input_shape, blob.ptr<float>());
            infer.set_input_tensor(input_tensor);
            infer.infer();
            auto c =chrono::steady_clock::now();

            // ---------- ③ 取结果 + 阈值过滤 + NMS ----------
            auto out_tensor = infer.get_output_tensor();
            const float* out = out_tensor.data<float>();
            int N = (int)out_tensor.get_shape()[2];

            vector<Detection> boxes;
            for (int i = 0; i < N; i++) {
                float best_s = 0.0f;
                int   best_c = 0;
                for (int cc = 0; cc < N_CLS; cc++) {
                    float s = out[(4 + cc) * N + i];
                    if (s > best_s) { best_s = s; best_c = cc; }
                }
                if (best_s > cfg.conf_thres) {
                    Detection d;
                    d.cx = out[0*N+i]; d.cy = out[1*N+i];
                    d.w  = out[2*N+i]; d.h  = out[3*N+i];
                    d.class_id = best_c; d.score = best_s;
                    boxes.push_back(d);
                }
            }

            vector<Detection> result;
            sort(boxes.begin(), boxes.end(), compareScore);
            for (int i = 0; i < (int)boxes.size(); i++) {
                bool keep = true;
                for (int j = 0; j < (int)result.size(); j++) {
                    if (boxes[i].class_id != result[j].class_id) continue;
                    if (iou(boxes[i], result[j]) > cfg.iou_thres) { keep = false; break; }
                }
                if (keep) result.push_back(boxes[i]);
            }
            auto d = chrono::steady_clock::now();

            // ---------- 只统计正式轮 ----------
            if (k >= WARMUP) {
                t_pre   += chrono::duration<double, milli>(b - a).count();
                t_infer += chrono::duration<double, milli>(c - b).count();
                t_post  += chrono::duration<double, milli>(d - c).count();
                box_pre  += boxes.size();
                box_post += result.size();
            }
        }

        // ---------- 打印 ----------
        double total = t_pre + t_infer + t_post;
        cout << "========== 推理基准（" << LOOPS << " 轮平均）==========" << endl;
        printf("  预处理  %8.3f ms  (%5.1f%%)\n", t_pre   / LOOPS, t_pre   / total * 100);
        printf("  推理    %8.3f ms  (%5.1f%%)   ← 关键\n", t_infer / LOOPS, t_infer / total * 100);
        printf("  后处理  %8.3f ms  (%5.1f%%)\n", t_post  / LOOPS, t_post  / total * 100);
        printf("  ----------------------------------------\n");
        printf("  合计    %8.3f ms  →  %.2f FPS\n", total / LOOPS, LOOPS * 1000.0 / total);
        printf("  NMS 前  %.2f 个/轮      NMS 后  %.2f 个/轮\n",
               (double)box_pre / LOOPS, (double)box_post / LOOPS);
        cout << "==========================================" << endl;

    } catch (const exception& e) {
        cout << "出错了：" << e.what() << endl;
        return 1;
    }
    return 0;
}
