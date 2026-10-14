// ==================================================================
//  车道线检测 —— 单图检测（OpenVINO 版）
//
//  参数从 config.json 读，改参数不用重新编译
// ==================================================================

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include "config.hpp"
using namespace std;

const char* CONFIG_PATH = "config.json";
const char* OUT_PATH    = "output/lane_result_ov.jpg";

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

        // 1. 读图
        cv::Mat img = cv::imread(cfg.image);
        if (img.empty()) { cout << "读图失败：" << cfg.image << endl; return 1; }
        cout << "配置：" << CONFIG_PATH << endl;
        cout << "图片：" << cfg.image << "   " << img.cols << " x " << img.rows << endl;
        cout << "模型：" << cfg.model_ov << "   输入 " << W_IN << " x " << H_IN << endl;

        // 2. 预处理
        cv::Mat blob = cv::dnn::blobFromImage(
            img, 1.0/255.0, cv::Size(W_IN, H_IN),
            cv::Scalar(), true, false);

        // 3. 加载模型
        ov::Core core;
        auto model    = core.read_model(cfg.model_ov);
        auto compiled = core.compile_model(model, "CPU");
        auto infer    = compiled.create_infer_request();

        // 4. 推理
        ov::Shape input_shape = { 1, 3, (size_t)H_IN, (size_t)W_IN };
        ov::Tensor input_tensor(ov::element::f32, input_shape, blob.ptr<float>());
        infer.set_input_tensor(input_tensor);
        infer.infer();

        // 5. 取结果
        auto out_tensor = infer.get_output_tensor();
        const float* out = out_tensor.data<float>();
        int N = (int)out_tensor.get_shape()[2];
        cout << "候选框数量：" << N << endl;

        // 6. 阈值过滤
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
        cout << "超过阈值的框：" << boxes.size() << " 个" << endl;

        // 7. NMS
        sort(boxes.begin(), boxes.end(), compareScore);
        vector<Detection> result;
        for (int i = 0; i < (int)boxes.size(); i++) {
            bool keep = true;
            for (int j = 0; j < (int)result.size(); j++) {
                if (boxes[i].class_id != result[j].class_id) continue;
                if (iou(boxes[i], result[j]) > cfg.iou_thres) { keep = false; break; }
            }
            if (keep) result.push_back(boxes[i]);
        }
        cout << "NMS 后剩下：" << result.size() << " 个框" << endl;

        // 8. 画框存图
        for (int i = 0; i < (int)result.size(); i++) {
            const Detection& de = result[i];
            double x1 = de.cx - de.w/2, y1 = de.cy - de.h/2;
            int X = (int)(x1 * img.cols / W_IN + 0.5);
            int Y = (int)(y1 * img.rows / H_IN + 0.5);
            int W = (int)(de.w * img.cols / W_IN + 0.5);
            int H = (int)(de.h * img.rows / H_IN + 0.5);

            cv::rectangle(img, cv::Rect(X, Y, W, H), cv::Scalar(0, 255, 0), 2);

            char label[64];
            snprintf(label, sizeof(label), "%s %.2f",
                     cfg.class_names[de.class_id].c_str(), de.score);
            int ty = (Y - 6 < 14) ? Y + 18 : Y - 6;
            cv::putText(img, label, cv::Point(X, ty),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite(OUT_PATH, img);
        cout << "已保存：" << OUT_PATH << endl;

    } catch (const exception& e) {
        cout << "出错了：" << e.what() << endl;
        return 1;
    }
    return 0;
}
