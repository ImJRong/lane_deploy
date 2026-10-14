// ==================================================================
//  车道线检测 —— 视频性能测试（OpenVINO 版）
//
//  跑视频、实时显示、统计平均帧率和平均框数，按 Q 提前退出
//  统计窗口固定（跳过前 SKIP_FRAMES 帧，统计 COUNT_FRAMES 帧），保证每次可比
//  换模型：改 MODEL_PATH 一行
// ==================================================================

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <chrono>
using namespace std;

// ---------- 配置区 ----------
const char* VIDEO_PATH = "data/lane_test.mp4";
//const char* MODEL_PATH = "models/lane_model.onnx";      // FP32
const char* MODEL_PATH = "models/lane_ov_int8.xml";   // 量化版

const int INPUT_WIDTH  = 640;
const int INPUT_HEIGHT = 640;

const char* CLASS_NAMES[] = { "lane" };
const int   NUM_CLASSES   = 1;

const float CONF_THRES = 0.25f;
const float IOU_THRES  = 0.5f;

const int SKIP_FRAMES  = 30;    // 前 30 帧预热，不计入统计
const int COUNT_FRAMES = 500;   // 统计接下来的 500 帧（固定，每次一样）

const char* WIN_NAME = "lane OV (press Q to quit)";
// ---------------------------

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
        // ---------- 打开视频 ----------
        cv::VideoCapture cap(VIDEO_PATH);
        if (!cap.isOpened()) { cout << "打开视频失败：" << VIDEO_PATH << endl; return 1; }

        int    total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
        double fps_src      = cap.get(cv::CAP_PROP_FPS);
        cout << "视频：" << VIDEO_PATH << endl;
        cout << "总帧数：" << total_frames << "   原始帧率：" << fps_src << endl;
        cout << "模型：" << MODEL_PATH << endl;
        cout << "统计窗口：跳过前 " << SKIP_FRAMES << " 帧，统计后 " << COUNT_FRAMES << " 帧" << endl;
        cout << "（按 Q 或 Esc 提前退出）\n" << endl;

        // ---------- 加载模型 ----------
        ov::Core core;
        auto model    = core.read_model(MODEL_PATH);
        auto compiled = core.compile_model(model, "CPU");
        auto infer    = compiled.create_infer_request();
        ov::Shape input_shape = { 1, 3, (size_t)INPUT_HEIGHT, (size_t)INPUT_WIDTH };

        // ---------- 逐帧处理 ----------
        cv::Mat frame;
        int       frame_idx = 0;    // 视频里的第几帧
        int       counted   = 0;    // 已计入统计的帧数
        double    sum_ms    = 0;    // 纯处理耗时累加
        long long box_pre   = 0;    // NMS 前框数累加
        long long box_post  = 0;    // NMS 后框数累加

        auto t_all_start = chrono::high_resolution_clock::now();

        while (true) {
            auto t_loop0 = chrono::high_resolution_clock::now();

            cap >> frame;
            if (frame.empty()) break;                 // 视频读完

            // ---- 1. 预处理 ----
            cv::Mat blob = cv::dnn::blobFromImage(
                frame, 1.0/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
                cv::Scalar(), true, false);

            // ---- 2. 推理 ----
            ov::Tensor input_tensor(ov::element::f32, input_shape, blob.ptr<float>());
            infer.set_input_tensor(input_tensor);
            infer.infer();

            // ---- 3. 取结果 + 阈值过滤 ----
            auto out_tensor = infer.get_output_tensor();
            const float* out = out_tensor.data<float>();
            int N = (int)out_tensor.get_shape()[2];

            vector<Detection> boxes;
            for (int i = 0; i < N; i++) {
                float best_s = 0.0f;
                int   best_c = 0;
                for (int cc = 0; cc < NUM_CLASSES; cc++) {
                    float s = out[(4 + cc) * N + i];
                    if (s > best_s) { best_s = s; best_c = cc; }
                }
                if (best_s > CONF_THRES) {
                    Detection d;
                    d.cx = out[0*N+i]; d.cy = out[1*N+i];
                    d.w  = out[2*N+i]; d.h  = out[3*N+i];
                    d.class_id = best_c; d.score = best_s;
                    boxes.push_back(d);
                }
            }

            // ---- 4. NMS ----
            vector<Detection> result;
            sort(boxes.begin(), boxes.end(), compareScore);
            for (int i = 0; i < (int)boxes.size(); i++) {
                bool keep = true;
                for (int j = 0; j < (int)result.size(); j++) {
                    if (boxes[i].class_id != result[j].class_id) continue;
                    if (iou(boxes[i], result[j]) > IOU_THRES) { keep = false; break; }
                }
                if (keep) result.push_back(boxes[i]);
            }

            auto t_proc_end = chrono::high_resolution_clock::now();
            double proc_ms = chrono::duration<double, milli>(t_proc_end - t_loop0).count();

            // ---- 5. 计入统计（只统计固定窗口内的帧）----
            if (frame_idx >= SKIP_FRAMES && counted < COUNT_FRAMES) {
                sum_ms   += proc_ms;
                box_pre  += boxes.size();
                box_post += result.size();
                counted++;
            }

            // ---- 6. 画框 ----
            for (int i = 0; i < (int)result.size(); i++) {
                const Detection& de = result[i];
                double x1 = de.cx - de.w/2, y1 = de.cy - de.h/2;
                int X = (int)(x1 * frame.cols / INPUT_WIDTH  + 0.5);
                int Y = (int)(y1 * frame.rows / INPUT_HEIGHT + 0.5);
                int W = (int)(de.w * frame.cols / INPUT_WIDTH  + 0.5);
                int H = (int)(de.h * frame.rows / INPUT_HEIGHT + 0.5);
                cv::rectangle(frame, cv::Rect(X, Y, W, H), cv::Scalar(0, 255, 0), 2);
            }

            char info[160];
            snprintf(info, sizeof(info), "frame %d   %.1f ms   %.1f FPS   box %d/%d",
                     frame_idx, proc_ms, 1000.0 / proc_ms,
                     (int)boxes.size(), (int)result.size());
            cv::putText(frame, info, cv::Point(15, 35),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

            // ---- 7. 显示 ----
            cv::imshow(WIN_NAME, frame);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) break;   // Q 或 Esc 退出

            frame_idx++;
            if (counted >= COUNT_FRAMES) break;                 // 统计够数就停
        }

        auto t_all_end = chrono::high_resolution_clock::now();
        double wall_s = chrono::duration<double, std::milli>(t_all_end - t_all_start).count() / 1000.0;

        // ---------- 统计 ----------
        cout << "\n=========== 视频性能测试（OpenVINO）===========" << endl;
        cout << "看过的帧数：" << frame_idx + 1 << "    计入统计：" << counted << endl;
        if (counted > 0) {
            printf("  纯处理    %7.2f ms/帧   →  %.2f FPS\n",
                   sum_ms / counted, 1000.0 / (sum_ms / counted));
            printf("  NMS 前    %7.2f 个/帧（共 %lld 个）\n",
                   (double)box_pre / counted, box_pre);
            printf("  NMS 后    %7.2f 个/帧（共 %lld 个）\n",
                   (double)box_post / counted, box_post);
        }
        printf("  端到端    %7.2f 秒   →  %.2f FPS（含读帧+画框+显示）\n",
               wall_s, (frame_idx + 1) / wall_s);
        cout << "=============================================" << endl;

    } catch (const exception& e) {
        cout << "出错了：" << e.what() << endl;
        return 1;
    }
    return 0;
}
