cmake_minimum_required(VERSION 3.10)
project(LaneDeploy)
set(CMAKE_CXX_STANDARD 17)

set(ORT_ROOT "$ENV{HOME}/onnxruntime-linux-x64-1.31.0")
set(OpenVINO_DIR "$ENV{HOME}/.local/lib/python3.14/site-packages/openvino/cmake")

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(Threads REQUIRED)

# ---------- 单图检测 ----------
add_executable(lane_ort src/lane_ort.cpp)
target_include_directories(lane_ort PRIVATE "${ORT_ROOT}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lane_ort PRIVATE "${ORT_ROOT}/lib/libonnxruntime.so" ${OpenCV_LIBS})
set_target_properties(lane_ort PROPERTIES BUILD_RPATH "${ORT_ROOT}/lib")

add_executable(lane_ov src/lane_ov.cpp)
target_include_directories(lane_ov PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lane_ov PRIVATE
    openvino::runtime openvino::frontend::onnx ${OpenCV_LIBS})

# ---------- 视频性能测试 ----------
add_executable(lane_fps_ort src/lane_fps_test_ort.cpp)
target_include_directories(lane_fps_ort PRIVATE "${ORT_ROOT}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lane_fps_ort PRIVATE "${ORT_ROOT}/lib/libonnxruntime.so" ${OpenCV_LIBS})
set_target_properties(lane_fps_ort PROPERTIES BUILD_RPATH "${ORT_ROOT}/lib")

add_executable(lane_fps_ov src/lane_fps_test_ov.cpp)
target_include_directories(lane_fps_ov PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lane_fps_ov PRIVATE
    openvino::runtime openvino::frontend::onnx ${OpenCV_LIBS})

# ---------- 逐层性能剖析 ----------
add_executable(lane_prof_ov src/lane_prof_ov.cpp)
target_include_directories(lane_prof_ov PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lane_prof_ov PRIVATE
    openvino::runtime openvino::frontend::onnx ${OpenCV_LIBS})

# ---------- 推理基准（单图循环 + 三段计时）----------
add_executable(lane_bench_ov src/lane_bench_ov.cpp)
target_include_directories(lane_bench_ov PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lane_bench_ov PRIVATE
    openvino::runtime openvino::frontend::onnx ${OpenCV_LIBS})

# ---------- 视频性能测试（多线程）----------
add_executable(lane_fps_mt_ov src/lane_fps_mt_ov.cpp)
target_include_directories(lane_fps_mt_ov PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lane_fps_mt_ov PRIVATE
    openvino::runtime openvino::frontend::onnx ${OpenCV_LIBS} Threads::Threads)
