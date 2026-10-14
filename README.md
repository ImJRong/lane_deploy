# 车道线检测模型 C++ 部署

自训练的 YOLOv8n 车道线模型（单类 lane，输入 640×640），在 WSL2 Ubuntu 上用 C++ 推理。
对比 ONNX Runtime 和 OpenVINO 两个引擎，覆盖 CPU 与核显两种设备、FP32 与 INT8 两种精度。

## 效果

![检测结果](output/lane_result_ort.jpg)

## 实测性能

环境 WSL2 Ubuntu、Intel i5-12500H，输入 640×640，conf=0.25、iou=0.5。
测试用 20 秒视频（599 帧），跳过前 30 帧预热，统计后 500 帧取平均。
「纯处理」指读帧之后的预处理、推理、后处理三段，不含视频解码和显示。

| 引擎 | 设备 | 精度 | 单帧耗时 | 帧率 | NMS 后框数 |
|---|---|---|---|---|---|
| ONNX Runtime | CPU | FP32 | 39.42 ms | 25.37 FPS | 7.77 |
| ONNX Runtime | CPU | INT8 | 60.42 ms | 16.55 FPS | 7.13 |
| OpenVINO | CPU | FP32 | 41.44 ms | 24.13 FPS | 7.77 |
| OpenVINO | CPU | INT8 | 27.37 ms | 36.54 FPS | 7.21 |
| OpenVINO | 核显 | FP32 | 14.37 ms | 69.6 FPS | 7.77 |
| OpenVINO | 核显 | INT8 | 18.66 ms | 53.6 FPS | 7.21 |

核显两行只有 OpenVINO，ONNX Runtime 在 WSL 里访问不到 Intel 核显。
耗时和帧率互为倒数，核显两行的耗时是按帧率折算的。核显数据连跑 3 遍取稳定值。

三条结论：

换设备最有效。OpenVINO 加 FP32，从 CPU 的 24.13 换到核显的 69.6，2.9 倍。模型和代码都没动，只改了配置里的 device。

量化不一定提速。同一个 FP32 模型转成 INT8 后：ONNX Runtime 在 CPU 上从 25.37 掉到 16.55，慢 35%；OpenVINO 在 CPU 上从 24.13 涨到 36.54，快 51%；OpenVINO 在核显上从 69.6 掉到 53.6，慢 23%。同一个动作，三个方向。

引擎不影响结果。FP32 下两个引擎的 NMS 后框数完全一致，都是 7.77，速度差 5%。量化后降到 7.13 和 7.21，约 8% 的精度损失，是量化带来的。

多线程读帧收益很小。把读帧拆到独立线程后，端到端从 31.14 涨到 31.83 FPS，只快 2.2%。

**核显上 INT8 为什么慢，做过排查，但没定位到原因。**&#8203; 试过的方法：

- 用 `get_profiling_info()` 取逐层耗时，只能算到算子执行层的差异，剩下的 GPU 侧开销拆不开
- 用分段计时逐段对比，但同一段代码在不同次运行里波动过大（同一行 `blobFromImage` 测出 1.9 ms 到 4.8 ms），这套计时在本机不可复现，数据不能用来归因
- 试过 OpenVINO 的 `queue_throttle` 属性，预处理耗时没有变化
- 把计时器从 `high_resolution_clock` 换成 `steady_clock`，结果同样不变

最后只保留了可以复现的总耗时差，原因一栏留空。

## 怎么跑

需要 OpenCV、ONNX Runtime 1.31.0、OpenVINO。路径在 `CMakeLists.txt` 里写死，换机器要改 `ORT_ROOT` 和 `OpenVINO_DIR`。

```bash
cmake . && make

./lane_fps_ort     # ONNX Runtime 版，跑视频测帧率
./lane_fps_ov      # OpenVINO 版，同上
./lane_ort         # ONNX Runtime 版，单图检测
./lane_ov          # OpenVINO 版，同上
```

模型和视频按下面的路径放好。跑视频时按 Q 或 Esc 退出。
`make` 还会编译出 `lane_fps_mt_ov`（多线程读帧）、`lane_bench_ov`（单图循环）、`lane_prof_ov`（逐层耗时剖析），这三个是调试用的。

## 配置

参数都写在 `config.json` 里，改完直接重跑，不用重新编译。

| 字段 | 说明 |
|---|---|
| `model_ort` / `model_ov` | 两个引擎各自的模型路径 |
| `device` | 推理设备，CPU 或 GPU |
| `image` / `video` | 单图版和视频版的输入 |
| `input_width` / `input_height` | 模型输入尺寸，必须和导出时一致 |
| `conf_thres` | 分数阈值，低于它的框丢掉 |
| `iou_thres` | NMS 的去重阈值 |
| `skip_frames` | 视频开头跳过几帧，避开热身 |
| `count_frames` | 统计多少帧，固定住才能横向比较 |
| `class_names` | 类别名，数量就是类别数 |

模型路径分两个，因为 ORT 读不了 .xml。共用一个字段的话，切换引擎时会直接报错。

## 文件说明

| 路径 | 内容 |
|---|---|
| src/lane_fps_test_ort.cpp | ONNX Runtime 版，跑视频统计帧率和框数 |
| src/lane_fps_test_ov.cpp | OpenVINO 版，同上 |
| src/lane_ort.cpp | ONNX Runtime 版，单图检测并存结果图 |
| src/lane_ov.cpp | OpenVINO 版，同上 |
| src/lane_fps_mt_ov.cpp | 读帧独立线程的版本 |
| src/lane_bench_ov.cpp | 单图循环 500 轮 |
| src/lane_prof_ov.cpp | 逐层耗时剖析 |
| src/config.hpp | 读配置的封装，定义 Config 结构体 |
| src/json.hpp | 读 JSON 用的库，单文件无依赖 |
| src/useless/ | 早期版本，不参与编译 |
| models/lane_model.onnx | 原始模型，FP32，两个引擎共用 |
| models/lane_ort_int8.onnx | ONNX Runtime 量化版 |
| models/lane_ov_int8.xml | OpenVINO 量化版，需要同目录的 .bin 文件 |
| data/lane_test.mp4 | 测试视频 |
| data/lane_test.jpg | 测试图片 |
| tools/quant_for_ov.py | 用 NNCF 把 ONNX 量化成 OpenVINO 格式 |
| config.json | 所有运行参数 |
| CMakeLists.txt | 编译配置 |

## 模型来源

在 Cityscapes 之外自采的 94 张车道线图片上，用 yolov8n 迁移学习训练，50 轮，单类。导出 ONNX 时 imgsz 必须和训练一致，都是 640。
