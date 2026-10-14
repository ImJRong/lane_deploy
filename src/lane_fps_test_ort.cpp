// ==================================================================
//  车道线检测 —— 视频性能测试（ONNX Runtime 版）
//
//  参数从 config.json 读，改参数不用重新编译
//  跑视频、实时显示、统计平均帧率和平均框数，按 Q 提前退出
// ==================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include "config.hpp"
using namespace std;

const char* CONFIG_PATH = "config.json";
const char* WIN_NAME    = "lane ORT (press Q to quit)";

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

        int    W_IN = cfg.input_width;
        int    H_IN = cfg.input_height;
        int    N_CLS = (int)cfg.class_names.size();

        // ---------- 打开视频 ----------
        cv::VideoCapture cap(cfg.video);
        if (!cap.isOpened()) { cout << "打开视频失败：" << cfg.video << endl; return 1; }

        int    total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
        double fps_src      = cap.get(cv::CAP_PROP_FPS);
        cout << "配置：" << CONFIG_PATH << endl;
        cout << "视频：" << cfg.video << endl;
        cout << "总帧数：" << total_frames << "   原始帧率：" << fps_src << endl;
        cout << "模型：" << cfg.model_ort << endl;
        cout << "输入尺寸：" << W_IN << " x " << H_IN << endl;
        cout << "统计窗口：跳过前 " << cfg.skip_frames << " 帧，统计后 " << cfg.count_frames << " 帧" << endl;
        cout << "（按 Q 或 Esc 提前退出）\n" << endl;

        // ---------- 加载模型 ----------
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "LaneFpsOrt");
        Ort::SessionOptions opt;
        Ort::Session session(env, ORT_TSTR(cfg.model_ort.c_str()), opt);

        const char* in_names[]  = { "images"  };
        const char* out_names[] = { "output0" };
        vector<int64_t> shape = { 1, 3, H_IN, W_IN };
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // ---------- 逐帧处理 ----------
        cv::Mat frame;
        int       frame_idx = 0;
        int       counted   = 0;
        double    sum_ms    = 0;
        long long box_pre   = 0;
        long long box_post  = 0;

        auto t_all_start = chrono::high_resolution_clock::now();

        while (true) {
            auto t_loop0 = chrono::high_resolution_clock::now();

            cap >> frame;
            if (frame.empty()) break;

            // ---- 1. 预处理 ----
            cv::Mat blob = cv::dnn::blobFromImage(
                frame, 1.0/255.0, cv::Size(W_IN, H_IN),
                cv::Scalar(), true, false);

            // ---- 2. 推理 ----
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                mem, blob.ptr<float>(), 1*3*H_IN*W_IN,
                shape.data(), shape.size());
            auto outputs = session.Run(Ort::RunOptions{nullptr},
                                       in_names, &input_tensor, 1, out_names, 1);

            // ---- 3. 取结果 + 阈值过滤 ----
            const float* out = outputs[0].GetTensorData<float>();
            int N = (int)outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];

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

            // ---- 4. NMS ----
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

            auto t_proc_end = chrono::high_resolution_clock::now();
            double proc_ms = chrono::duration<double, milli>(t_proc_end - t_loop0).count();

            // ---- 5. 计入统计 ----
            if (frame_idx >= cfg.skip_frames && counted < cfg.count_frames) {
                sum_ms   += proc_ms;
                box_pre  += boxes.size();
                box_post += result.size();
                counted++;
            }

            // ---- 6. 画框 ----
            for (int i = 0; i < (int)result.size(); i++) {
                const Detection& de = result[i];
                double x1 = de.cx - de.w/2, y1 = de.cy - de.h/2;
                int X = (int)(x1 * frame.cols / W_IN + 0.5);
                int Y = (int)(y1 * frame.rows / H_IN + 0.5);
                int W = (int)(de.w * frame.cols / W_IN + 0.5);
                int H = (int)(de.h * frame.rows / H_IN + 0.5);
                cv::rectangle(frame, cv::Rect(X, Y, W, H), cv::Scalar(0, 0, 255), 2);
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
            if (key == 'q' || key == 'Q' || key == 27) break;

            frame_idx++;
            if (counted >= cfg.count_frames) break;
        }

        auto t_all_end = chrono::high_resolution_clock::now();
        double wall_s = chrono::duration<double, std::milli>(t_all_end - t_all_start).count() / 1000.0;

        // ---------- 统计 ----------
        cout << "\n=========== 视频性能测试（ONNX Runtime）===========" << endl;
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
        cout << "================================================" << endl;

    } catch (const exception& e) {
        cout << "出错了：" << e.what() << endl;
        return 1;
    }
    return 0;
}
