// ==================================================================
//  车道线检测 —— 视频性能测试（OpenVINO 版 · 多线程）
//
//  跟单线程版的区别：读帧单独跑一个线程，主线程只管推理。
//
//    读帧线程：  cap.read() → 塞进队列
//    主线程：    从队列拿 → 预处理 → 推理 → 后处理 → 画框 → 显示
//
//  队列有上限，满了读帧线程就等；空了主线程就等。
//  参数从 config.json 读。
// ==================================================================

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include "config.hpp"
using namespace std;

const char* CONFIG_PATH = "config.json";
const char* WIN_NAME    = "lane OV MT (press Q to quit)";

const int MAX_QUEUE = 4;      // 队列最多攒几帧，防止读得太快把内存撑爆

// ---------- 线程间共享的队列 ----------
static queue<cv::Mat> g_queue;
static mutex          g_mtx;
static condition_variable g_cv_not_full;    // 队列没满，读帧线程可以继续塞
static condition_variable g_cv_not_empty;   // 队列有货，主线程可以拿

// ---------- 读帧线程 ----------
static void reader_thread(cv::VideoCapture* cap, atomic<bool>* finished) {
    while (true) {
        cv::Mat frame;
        if (!cap->read(frame) || frame.empty()) break;

        {
            unique_lock<mutex> lock(g_mtx);
            // 队列满了就等，等主线程腾出位置
            // 注意：谓词里必须带上 finished —— 否则主线程叫停时，只要队列还是满的，这里就会一直等下去
            g_cv_not_full.wait(lock, [&] {
                return (int)g_queue.size() < MAX_QUEUE || finished->load();
            });
            if (finished->load()) return;      // 主线程叫停，直接退出线程
            g_queue.push(frame);
        }
        g_cv_not_empty.notify_one();
    }
    finished->store(true);
    g_cv_not_empty.notify_one();      // 叫醒可能还在等的主线程
}


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

        // ---------- 打开视频 ----------
        cv::VideoCapture cap(cfg.video);
        if (!cap.isOpened()) { cout << "打开视频失败：" << cfg.video << endl; return 1; }

        int    total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
        double fps_src      = cap.get(cv::CAP_PROP_FPS);
        cout << "配置：" << CONFIG_PATH << "   【多线程版】" << endl;
        cout << "视频：" << cfg.video << endl;
        cout << "总帧数：" << total_frames << "   原始帧率：" << fps_src << endl;
        cout << "模型：" << cfg.model_ov << endl;
        cout << "输入尺寸：" << W_IN << " x " << H_IN << endl;
        cout << "统计窗口：跳过前 " << cfg.skip_frames << " 帧，统计后 " << cfg.count_frames << " 帧" << endl;
        cout << "（按 Q 或 Esc 提前退出）\n" << endl;

        // ---------- 加载模型 ----------
        ov::Core core;
        auto model    = core.read_model(cfg.model_ov);
        auto compiled = core.compile_model(model, cfg.device);
        auto infer    = compiled.create_infer_request();
        ov::Shape input_shape = { 1, 3, (size_t)H_IN, (size_t)W_IN };

        // ---------- 启动读帧线程 ----------
        atomic<bool> reader_done(false);
        thread reader(reader_thread, &cap, &reader_done);

        // ---------- 主循环 ----------
        int       frame_idx = 0;
        int       counted   = 0;
        double    sum_ms    = 0;
        long long box_pre   = 0;
        long long box_post  = 0;
        bool      quit      = false;

        auto t_all_start = chrono::high_resolution_clock::now();

        while (true) {
            cv::Mat frame;
            {
                unique_lock<mutex> lock(g_mtx);
                g_cv_not_empty.wait(lock, [&] { return !g_queue.empty() || reader_done.load(); });
                if (g_queue.empty() && reader_done.load()) break;   // 读完了，退出
                frame = g_queue.front();      // cv::Mat 是引用计数，这里只是浅拷贝
                g_queue.pop();
            }
            g_cv_not_full.notify_one();       // 告诉读帧线程：有位置了

            auto t_loop0 = chrono::high_resolution_clock::now();

            // ---- 预处理 ----
            cv::Mat blob = cv::dnn::blobFromImage(
                frame, 1.0/255.0, cv::Size(W_IN, H_IN),
                cv::Scalar(), true, false);

            // ---- 推理 ----
            ov::Tensor input_tensor(ov::element::f32, input_shape, blob.ptr<float>());
            infer.set_input_tensor(input_tensor);
            infer.infer();

            // ---- 取结果 + 阈值过滤 ----
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

            // ---- NMS ----
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

            // ---- 计入统计 ----
            if (frame_idx >= cfg.skip_frames && counted < cfg.count_frames) {
                sum_ms   += proc_ms;
                box_pre  += boxes.size();
                box_post += result.size();
                counted++;
            }

            // ---- 画框 ----
            for (int i = 0; i < (int)result.size(); i++) {
                const Detection& de = result[i];
                double x1 = de.cx - de.w/2, y1 = de.cy - de.h/2;
                int X = (int)(x1 * frame.cols / W_IN + 0.5);
                int Y = (int)(y1 * frame.rows / H_IN + 0.5);
                int W = (int)(de.w * frame.cols / W_IN + 0.5);
                int H = (int)(de.h * frame.rows / H_IN + 0.5);
                cv::rectangle(frame, cv::Rect(X, Y, W, H), cv::Scalar(0, 255, 0), 2);
            }

            char info[160];
            snprintf(info, sizeof(info), "MT  frame %d   %.1f ms   %.1f FPS   box %d/%d",
                     frame_idx, proc_ms, 1000.0 / proc_ms,
                     (int)boxes.size(), (int)result.size());
            cv::putText(frame, info, cv::Point(15, 35),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

            // ---- 显示 ----
            cv::imshow(WIN_NAME, frame);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) { quit = true; break; }

            frame_idx++;
            if (counted >= cfg.count_frames) break;
        }

        auto t_all_end = chrono::high_resolution_clock::now();
        double wall_s = chrono::duration<double, std::milli>(t_all_end - t_all_start).count() / 1000.0;

        // ---------- 收尾：不管什么原因退出，都要叫醒读帧线程 ----------
        // 主线程退出有三种情况：统计够帧数、按 Q、视频读完。
        // 前两种发生时读帧线程可能还在跑，必须叫醒它，否则 join 会永久阻塞。
        reader_done.store(true);
        g_cv_not_full.notify_all();
        g_cv_not_empty.notify_all();
        reader.join();

        // ---------- 统计 ----------
        cout << "\n=========== 视频性能测试（OpenVINO · 多线程）===========" << endl;
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
        cout << "======================================================" << endl;

    } catch (const exception& e) {
        cout << "出错了：" << e.what() << endl;
        return 1;
    }
    return 0;
}
