// ==================================================================
//  C++ + ONNX Runtime + OpenCV 跑 YOLO 推理（Linux 版）
//
//  跟手搓版的流程完全一样（9 步），只是把「读图 / 预处理 / 画框 / 存图」
//  全换成 OpenCV 来做 —— 代码短了一大截。
//
//  ① 读图片     cv::imread
//  ② 预处理     cv::dnn::blobFromImage（缩放 + RGB + NCHW + 归一化，一行搞定）
//  ③ 加载模型   ONNX Runtime
//  ④ 包成 Tensor
//  ⑤ 跑推理
//  ⑥ 取结果
//  ⑦ 阈值过滤   每个框取最高分类别
//  ⑧ NMS 去重
//  ⑨ 画框存图   cv::rectangle + cv::putText + cv::imwrite
// ==================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <onnxruntime_cxx_api.h>
using namespace std;

// 图片和模型都放在程序同目录，直接用文件名
const char* IMAGE_PATH = "test.jpg";
const char* MODEL_PATH = "best.onnx";

const int INPUT_WIDTH  = 960;
const int INPUT_HEIGHT = 960;

// 模型的 5 个类别（顺序必须和训练时一致）
const char* CLASS_NAMES[] = { "car", "person", "truck", "bus", "traffic light" };
const int   NUM_CLASSES   = 5;


struct Detection{
    double cx, cy, w, h;
    int    class_id;
    double score;
};

// 算两个框的重叠程度：交集面积 ÷ 并集面积（0~1）
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

// 给 sort 用：分数高的排前面
bool compareScore(const Detection& a, const Detection& b) {
    return a.score > b.score;
}


int main() {
    try {
        // ==========================================================
        //  1. 读图片（OpenCV 版，一行）
        // ==========================================================
        cv::Mat img = cv::imread(IMAGE_PATH);
        if (img.empty()) {
            cout << "读图失败：" << IMAGE_PATH << endl;
            return 1;
        }
        cout << "原图尺寸：" << img.cols << " x " << img.rows << endl;

        // ==========================================================
        //  2. 预处理（OpenCV 一行搞定四件事）
        //
        //  blobFromImage 帮我们做了：
        //    · 缩放到 960×960
        //    · BGR → RGB（swapRB = true）
        //    · 归一化 ÷255（scale = 1/255）
        //    · 排成 NCHW 的连续 float 数组（blob）
        // ==========================================================
        cv::Mat blob = cv::dnn::blobFromImage(
            img,
            1.0 / 255.0,                           // 缩放系数：÷255
            cv::Size(INPUT_WIDTH, INPUT_HEIGHT),   // 目标尺寸
            cv::Scalar(),                          // 要减的均值（不减）
            true,                                  // swapRB：BGR → RGB
            false);                                // crop：不裁剪
        cout << "预处理完成，blob = " << blob.size[0] << "×" << blob.size[1]
             << "×" << blob.size[2] << "×" << blob.size[3] << endl;

        // ==========================================================
        //  3. 加载模型
        // ==========================================================
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YoloOnnx");
        Ort::SessionOptions session_options;
        Ort::Session session(env, ORT_TSTR(MODEL_PATH), session_options);

        // ==========================================================
        //  4. 包成 Tensor（数据直接来自 blob，不用自己造数组了）
        // ==========================================================
        vector<int64_t> input_shape = { 1, 3, INPUT_HEIGHT, INPUT_WIDTH };
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            blob.ptr<float>(),                          // 数据指针
            1 * 3 * INPUT_HEIGHT * INPUT_WIDTH,         // 元素个数
            input_shape.data(),
            input_shape.size());

        // ==========================================================
        //  5. 跑推理
        // ==========================================================
        const char* input_names[]  = { "images"  };
        const char* output_names[] = { "output0" };

        cout << "开始推理..." << endl;
        auto outputs = session.Run(
            Ort::RunOptions{ nullptr },
            input_names, &input_tensor, 1,
            output_names, 1);
        cout << "推理完成！" << endl;

        // ==========================================================
        //  6. 取结果
        //
        //  输出 [1, 4+C, N]：前 4 行是 cx,cy,w,h（中心点 + 宽高），
        //  后 C 行是各类别分数。第 i 个框第 j 个值 = data[j * N + i]
        // ==========================================================
        const float* output_data = outputs[0].GetTensorData<float>();
        auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int num_boxes = (int)output_shape[2];

        cout << "候选框数量：" << num_boxes << endl;

        // ==========================================================
        //  7. 阈值过滤：每个框只取最高分类别，过阈值才留
        // ==========================================================
        vector<Detection> detectionBox;

        for (int i = 0; i < num_boxes; i++) {
            int   best_c = 0;
            float best_s = 0.0f;
            for (int c = 0; c < NUM_CLASSES; c++) {
                float score = output_data[(4 + c) * num_boxes + i];
                if (score > best_s) {
                    best_s = score;
                    best_c = c;
                }
            }
            if (best_s > 0.25f) {
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
        cout << "超过阈值的框：" << detectionBox.size() << " 个" << endl;

        // ==========================================================
        //  8. NMS 去重：排序 → 逐个跟已留的同类框比 IoU，>0.5 就丢
        // ==========================================================
        sort(detectionBox.begin(), detectionBox.end(), compareScore);

        vector<Detection> result;
        int n = (int)detectionBox.size();
        for (int i = 0; i < n; i++) {
            bool keep = true;
            int m = (int)result.size();
            for (int j = 0; j < m; j++) {
                if (detectionBox[i].class_id != result[j].class_id) continue;
                if (iou(detectionBox[i], result[j]) > 0.4) {
                    keep = false;
                    break;
                }
            }
            if (keep) result.push_back(detectionBox[i]);
        }
        cout << "NMS 后剩下：" << result.size() << " 个框" << endl;

        // ==========================================================
        //  9. 画框存图（OpenCV：画框 + 写类别名和置信度）
        //
        //  ⚠️ OpenCV 的颜色是 BGR 顺序（不是 RGB）
        // ==========================================================
        const cv::Scalar COLORS[5] = {
            cv::Scalar(  0,   0, 255),   // car           红
            cv::Scalar(  0, 255,   0),   // person        绿
            cv::Scalar(  0, 255, 255),   // truck         黄
            cv::Scalar(255, 160,   0),   // bus           蓝
            cv::Scalar(255,   0, 255),   // traffic light 品红
        };

        for (int i = 0; i < (int)result.size(); i++) {
            const Detection& d = result[i];

            // 中心点 + 宽高 → 左上角（还是 960 尺度）
            double x1 = d.cx - d.w / 2;
            double y1 = d.cy - d.h / 2;

            // 缩回原图尺寸（x、y 比例不同，先乘后除）
            int X = (int)(x1 * img.cols / INPUT_WIDTH  + 0.5);
            int Y = (int)(y1 * img.rows / INPUT_HEIGHT + 0.5);
            int W = (int)(d.w  * img.cols / INPUT_WIDTH  + 0.5);
            int H = (int)(d.h  * img.rows / INPUT_HEIGHT + 0.5);

            // 画框
            cv::rectangle(img, cv::Rect(X, Y, W, H), COLORS[d.class_id], 2);

            // 写字：类别名 + 置信度
            char label[64];
            snprintf(label, sizeof(label), "%s %.2f", CLASS_NAMES[d.class_id], d.score);
            int ty = (Y - 6 < 14) ? Y + 18 : Y - 6;   // 太靠顶就写到框里面，免得写出去
            cv::putText(img, label, cv::Point(X, ty),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, COLORS[d.class_id], 2);
        }

        // 存图
        cv::imwrite("test_result.jpg", img);
        cout << "已保存画框结果：test_result.jpg" << endl;

    } catch (const exception& e) {
        cout << "出错了：" << e.what() << endl;
        return 1;
    }
    return 0;
}
