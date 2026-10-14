// ==================================================================
//  车道线检测 —— 逐层性能剖析（OpenVINO 版）
//
//  用 ov::enable_profiling(true) 让 GPU/CPU 自己上报每一层的耗时，
//  绕开"CPU 侧打时间戳"在 GPU 上归属错乱的问题。
//
//  输出：
//    ① 最耗时的前 15 层（层名 / 算子类型 / 执行类型 / 耗时）
//    ② 按算子类型汇总（哪些类型的层占了大部分时间）
//
//  exec_type 那一列特别有用：它显示这层实际用什么内核跑的
//    - FP32 模型：会出现类似 jit_avx2_FP32 / ocl：xxx 的字样
//    - INT8 模型：会出现 u8s8 / i8i8 之类的字样
//  拿这个就能确认"量化到底生效了没有"。
// ==================================================================

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <cstdio>
#include "config.hpp"
using namespace std;

const char* CONFIG_PATH = "config.json";

struct LayerStat {
    string name;
    string type;
    string exec;
    double us;          // 微秒
};

int main() {
    try {
        Config cfg = load_config(CONFIG_PATH);

        int W_IN  = cfg.input_width;
        int H_IN  = cfg.input_height;

        cv::Mat img = cv::imread(cfg.image);
        if (img.empty()) { cout << "读图失败：" << cfg.image << endl; return 1; }

        cout << "模型：" << cfg.model_ov << endl;
        cout << "设备：" << cfg.device << "   输入 " << W_IN << " x " << H_IN << "\n" << endl;

        // ---------- 编译时开启 profiling ----------
        ov::Core core;
        auto model = core.read_model(cfg.model_ov);
        auto compiled = core.compile_model(model, cfg.device, ov::enable_profiling(true));
        auto infer = compiled.create_infer_request();

        ov::Shape input_shape = { 1, 3, (size_t)H_IN, (size_t)W_IN };

        // 预处理（单独做，不参与 profiling）
        cv::Mat blob = cv::dnn::blobFromImage(
            img, 1.0/255.0, cv::Size(W_IN, H_IN), cv::Scalar(), true, false);
        ov::Tensor input_tensor(ov::element::f32, input_shape, blob.ptr<float>());

        // 热身几轮，让 kernel 编译和缓存都就位
        for (int i = 0; i < 10; i++) {
            infer.set_input_tensor(input_tensor);
            infer.infer();
        }

        // ---------- 正式一轮，取 profiling ----------
        infer.set_input_tensor(input_tensor);
        infer.infer();

        auto infos = infer.get_profiling_info();

        vector<LayerStat> layers;
        map<string, pair<int, double>> by_type;     // 算子类型 -> (层数, 总耗时)
        double total_us = 0;

        for (auto& p : infos) {
            LayerStat s;
            s.name = p.node_name;
            s.type = p.node_type;
            s.exec = p.exec_type;
            s.us   = (double)p.real_time.count();
            layers.push_back(s);
            total_us += s.us;
            by_type[s.type].first  += 1;
            by_type[s.type].second += s.us;
        }

        printf("共 %d 层，profiling 总耗时 %.3f ms\n\n", (int)layers.size(), total_us / 1000.0);

        // ---------- ① 最耗时的前 15 层 ----------
        sort(layers.begin(), layers.end(),
             [](const LayerStat& a, const LayerStat& b) { return a.us > b.us; });

        printf("%-28s %-16s %-14s %10s\n", "层名", "算子类型", "执行类型", "耗时(us)");
        printf("----------------------------------------------------------------------------------\n");
        int top = (int)layers.size() < 15 ? (int)layers.size() : 15;
        for (int i = 0; i < top; i++) {
            const LayerStat& s = layers[i];
            string nm = s.name.size() > 26 ? s.name.substr(0, 26) + ".." : s.name;
            string ty = s.type.size() > 14 ? s.type.substr(0, 14) : s.type;
            string ex = s.exec.size() > 12 ? s.exec.substr(0, 12) : s.exec;
            printf("%-28s %-16s %-14s %10.1f\n", nm.c_str(), ty.c_str(), ex.c_str(), s.us);
        }

        // ---------- ② 按算子类型汇总 ----------
        vector<pair<string, pair<int, double>>> vt(by_type.begin(), by_type.end());
        sort(vt.begin(), vt.end(),
             [](const auto& a, const auto& b) { return a.second.second > b.second.second; });

        printf("\n%-18s %8s %14s %9s\n", "算子类型", "层数", "合计(us)", "占比");
        printf("--------------------------------------------------------\n");
        for (auto& kv : vt) {
            printf("%-18s %8d %14.1f %8.1f%%\n",
                   kv.first.c_str(), kv.second.first, kv.second.second,
                   kv.second.second / total_us * 100);
        }

    } catch (const exception& e) {
        cout << "出错了：" << e.what() << endl;
        return 1;
    }
    return 0;
}
