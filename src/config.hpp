// ==================================================================
//  config.hpp —— 读 config.json 的小工具
//
//  用法：
//      #include "config.hpp"
//      Config cfg = load_config("config.json");
//      cfg.model_ort / cfg.conf_thres / ...
//
//  模型分两个字段：ORT 版读 model_ort，OpenVINO 版读 model_ov。
//  这样换模型时互不影响（ORT 只认 .onnx，OpenVINO 两种都认）。
// ==================================================================

#pragma once

#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

struct Config {
    std::string model_ort;                      // ONNX Runtime 用的模型
    std::string model_ov;                       // OpenVINO 用的模型

    std::string image;                          // 单图测试用的图片
    std::string video;                          // 视频测试用的视频

    int input_width  = 640;
    int input_height = 640;

    float conf_thres = 0.25f;
    float iou_thres  = 0.5f;

    int skip_frames  = 30;
    int count_frames = 500;

    std::vector<std::string> class_names;
};

inline Config load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "读不到配置文件：" << path << std::endl;
        std::exit(1);
    }

    nlohmann::json j;
    f >> j;

    Config c;
    c.model_ort    = j.value("model_ort",    c.model_ort);
    c.model_ov     = j.value("model_ov",     c.model_ov);
    c.image        = j.value("image",        c.image);
    c.video        = j.value("video",        c.video);
    c.input_width  = j.value("input_width",  c.input_width);
    c.input_height = j.value("input_height", c.input_height);
    c.conf_thres   = j.value("conf_thres",   c.conf_thres);
    c.iou_thres    = j.value("iou_thres",    c.iou_thres);
    c.skip_frames  = j.value("skip_frames",  c.skip_frames);
    c.count_frames = j.value("count_frames", c.count_frames);

    if (j.contains("class_names")) {
        c.class_names = j["class_names"].get<std::vector<std::string>>();
    }

    if (c.class_names.empty()) {
        std::cerr << "配置里没有 class_names" << std::endl;
        std::exit(1);
    }

    return c;
}
