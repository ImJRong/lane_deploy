// ==================================================================
//  C++ + OpenVINO + OpenCV 跑 YOLO 推理（性能测试版）
//
//  跟 ONNX Runtime 版流程完全一样，只换了推理引擎。
//  三段计时：① 预处理 ② 推理 ③ 后处理
// ==================================================================

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <chrono>
using namespace std;

const char* IMAGE_PATH = "test.jpg";  // 测试图片
const char* MODEL_PATH = "best_int8.onnx";
//const char* MODEL_PATH = "best_int8.onnx";

const int INPUT_WIDTH  = 960;
const int INPUT_HEIGHT = 960;

// 性能测试参数
const int WARMUP = 10;     // 前几轮热身：第一轮要建缓存/分配内存，比后面慢，不能算进平均
const int LOOPS  = 50;     // 正式统计 50 轮

const char* CLASS_NAMES[] = { "car", "person", "truck", "bus", "traffic light" };
const int   NUM_CLASSES   = 5;


struct Detection{
    double cx, cy, w, h;
    int    class_id;
    double score;
};

float iou(const Detection& a, const Detection& b) {
    float ax1 = a.cx - a.w/2, ay1 = a.cy - a.h/2, ax2 = a.cx + a.w/2, ay2 = a.cy + a.h/2;
    float bx1 = b.cx - b.w/2, by1 = b.cy - b.h/2, bx2 = b.cx + b.w/2, by2 = b.cy + b.h/2;

    float inter_w = min(ax2, bx2) - max(ax1, bx1);
    float inter_h = min(ay2, by2) - max(ay1, by1);
    if (inter_w < 0) inter_w = 0;
    if (inter_h < 0) inter_h = 0;
    float inter_area = inter_w * inter_h;

    float union_area = a.w * a.h + b.w * b.h - inter_area;
    return inter_area / union_area;
}

bool compareScore(const Detection& a, const Detection& b) {
    return a.score > b.score;
}


int main() {
    try {
        // ==========================================================
        //  1. 读图片
        // ==========================================================
        cv::Mat img = cv::imread(IMAGE_PATH);
        if (img.empty()) {
            cout << "读图失败：" << IMAGE_PATH << endl;
            return 1;
        }
        cout << "原图尺寸：" << img.cols << " x " << img.rows << endl;
        cout << "模型文件：" << MODEL_PATH << endl;

        // ==========================================================
        //  2. 加载模型（★ 绝对不要放进循环）
        //     OpenVINO 分三步：读模型 → 针对 CPU 编译 → 创建推理请求
        //     compile_model 第一次要几秒（在编译），但只做一次
        // ==========================================================
        ov::Core core;
        auto model    = core.read_model(MODEL_PATH);
        auto compiled = core.compile_model(model, "CPU");
        auto infer    = compiled.create_infer_request();
        cout << "模型加载完成" << endl;

        ov::Shape input_shape = { 1, 3, (size_t)INPUT_HEIGHT, (size_t)INPUT_WIDTH };

        // ==========================================================
        //  3. 性能测试主循环
        // ==========================================================
        double t_pre = 0, t_infer = 0, t_post = 0;   // 三段耗时累加（毫秒）
        vector<Detection> result;                    // 最后一轮的结果，留给画框用

        cout << "\n开始跑 " << (WARMUP + LOOPS) << " 轮（前 " << WARMUP << " 轮热身，不计入统计）...\n" << endl;

        for (int k = 0; k < WARMUP + LOOPS; k++) {

            // ---------- ① 预处理 ----------
            auto a = chrono::high_resolution_clock::now();

            cv::Mat blob = cv::dnn::blobFromImage(
                img,
                1.0 / 255.0,                           // 缩放系数：÷255
                cv::Size(INPUT_WIDTH, INPUT_HEIGHT),   // 目标尺寸
                cv::Scalar(),                          // 要减的均值（不减）
                true,                                  // swapRB：BGR → RGB
                false);                                // crop：不裁剪

            auto b = chrono::high_resolution_clock::now();

            // ---------- ② 包 Tensor + 推理 ----------
            ov::Tensor input_tensor(ov::element::f32, input_shape, blob.ptr<float>());
            infer.set_input_tensor(input_tensor);
            infer.infer();

            auto c = chrono::high_resolution_clock::now();

            // ---------- ③ 取结果 + 阈值过滤 + NMS ----------
            auto output_tensor = infer.get_output_tensor();
            const float* output_data = output_tensor.data<float>();
            ov::Shape output_shape = output_tensor.get_shape();
            int num_boxes = (int)output_shape[2];

            // 阈值过滤：每个框只取最高分类别
            vector<Detection> detectionBox;
            for (int i = 0; i < num_boxes; i++) {
                int   best_c = 0;
                float best_s = 0.0f;
                for (int c2 = 0; c2 < NUM_CLASSES; c2++) {
                    float score = output_data[(4 + c2) * num_boxes + i];
                    if (score > best_s) {
                        best_s = score;
                        best_c = c2;
                    }
                }
                if (best_s > 0.2f) {
                    Detection det;
                    det.cx       = output_data[0 * num_boxes + i];
                    det.cy       = output_data[1 * num_boxes + i];
                    det.w        = output_data[2 * num_boxes + i];
                    det.h        = output_data[3 * num_boxes + i];
                    det.class_id = best_c;
                    det.score    = best_s;
                    detectionBox.push_back(det);
                }
            }

            // NMS 去重
            sort(detectionBox.begin(), detectionBox.end(), compareScore);

            result.clear();                          // ★ 每轮必须清空，否则会一轮轮累加
            int n = (int)detectionBox.size();
            for (int i = 0; i < n; i++) {
                bool keep = true;
                int m = (int)result.size();
                for (int j = 0; j < m; j++) {
                    if (detectionBox[i].class_id != result[j].class_id) continue;
                    if (iou(detectionBox[i], result[j]) > 0.5) {
                        keep = false;
                        break;
                    }
                }
                if (keep) result.push_back(detectionBox[i]);
            }

            auto d = chrono::high_resolution_clock::now();

            // ---------- 只有正式轮才计入统计（热身轮跳过）----------
            if (k >= WARMUP) {
                t_pre   += chrono::duration<double, milli>(b - a).count();
                t_infer += chrono::duration<double, milli>(c - b).count();
                t_post  += chrono::duration<double, milli>(d - c).count();
            }
        }

        // ==========================================================
        //  4. 打印结果
        // ==========================================================
        double total = t_pre + t_infer + t_post;

        cout << "================ 性能测试结果（" << LOOPS << " 轮平均）================\n";
        printf("  预处理    %7.2f ms   (%5.1f%%)\n", t_pre   / LOOPS, t_pre   / total * 100);
        printf("  推理      %7.2f ms   (%5.1f%%)\n", t_infer / LOOPS, t_infer / total * 100);
        printf("  后处理    %7.2f ms   (%5.1f%%)\n", t_post  / LOOPS, t_post  / total * 100);
        printf("  ----------------------------------------------\n");
        printf("  合计      %7.2f ms   →   %.2f FPS\n", total / LOOPS, LOOPS * 1000.0 / total);
        cout << "================================================================" << endl;

        // ==========================================================
        //  5. 画框存图（循环外，用最后一轮的 result）
        // ==========================================================
        const cv::Scalar COLORS[5] = {
            cv::Scalar(  0,   0, 255),   // car           红
            cv::Scalar(  0, 255,   0),   // person        绿
            cv::Scalar(  0, 255, 255),   // truck         黄
            cv::Scalar(255, 160,   0),   // bus           蓝
            cv::Scalar(255,   0, 255),   // traffic light 品红
        };

        cout << "\nNMS 后剩下：" << result.size() << " 个框" << endl;

        for (int i = 0; i < (int)result.size(); i++) {
            const Detection& d = result[i];

            double x1 = d.cx - d.w / 2;
            double y1 = d.cy - d.h / 2;

            int X = (int)(x1 * img.cols / INPUT_WIDTH  + 0.5);
            int Y = (int)(y1 * img.rows / INPUT_HEIGHT + 0.5);
            int W = (int)(d.w  * img.cols / INPUT_WIDTH  + 0.5);
            int H = (int)(d.h  * img.rows / INPUT_HEIGHT + 0.5);

            cv::rectangle(img, cv::Rect(X, Y, W, H), COLORS[d.class_id], 2);

            char label[64];
            snprintf(label, sizeof(label), "%s %.2f", CLASS_NAMES[d.class_id], d.score);
            int ty = (Y - 6 < 14) ? Y + 18 : Y - 6;
            cv::putText(img, label, cv::Point(X, ty),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, COLORS[d.class_id], 2);
        }

        cv::imwrite("test_result_ov.jpg", img);
        cout << "已保存画框结果：test_result_ov.jpg" << endl;

    } catch (const exception& e) {
        cout << "出错了：" << e.what() << endl;
        return 1;
    }
    return 0;
}
