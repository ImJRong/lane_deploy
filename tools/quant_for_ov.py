# onnx(FP32) → OpenVINO int8（用 Intel 的 NNCF）
import glob, os
import numpy as np
import cv2
import openvino as ov
import nncf

FP32    = "/home/imrong/lane_deploy/models/lane_model.onnx"          # 脚本1 的 FP32
OUT_XML = "/home/imrong/lane_deploy/models/lane_ov_int8.xml"
CALIB   = "/mnt/d/yolo-practice/data/lane/images"              # 校准图（不用拷）
SIZE    = 640

# --- 数据加载 ---
files = sorted(glob.glob(os.path.join(CALIB, "*.jpg")))
print("校准图片数:", len(files))
if not files:
    raise SystemExit("找不到校准图片，检查 CALIB")

def transform(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SIZE, SIZE))            # ← 注意：这里也是拉伸
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, 0)

dataset = nncf.Dataset(files, transform)

# --- 量化 ---
core = ov.Core()
model = core.read_model(FP32)
print("FP32 已读入")

quantized = nncf.quantize(model, dataset, subset_size=len(files))
ov.save_model(quantized, OUT_XML)
print("已保存:", OUT_XML, "+ .bin")


//运行：  python3 tools/quant_for_ov.py