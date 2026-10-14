cmake_minimum_required(VERSION 3.10)
project(LaneDeploy)
set(CMAKE_CXX_STANDARD 17)

set(ORT_ROOT "$ENV{HOME}/onnxruntime-linux-x64-1.31.0")
find_package(OpenCV REQUIRED)

# ---- ORT 版 ----
add_executable(lane_ort src/lane_ort.cpp)
target_include_directories(lane_ort PRIVATE "${ORT_ROOT}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lane_ort PRIVATE "${ORT_ROOT}/lib/libonnxruntime.so" ${OpenCV_LIBS})
set_target_properties(lane_ort PROPERTIES BUILD_RPATH "${ORT_ROOT}/lib")

# ---- OpenVINO 版----
set(OpenVINO_DIR "$ENV{HOME}/.local/lib/python3.14/site-packages/openvino/cmake")
find_package(OpenVINO REQUIRED)
add_executable(lane_ov src/lane_ov.cpp)
target_include_directories(lane_ov PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lane_ov PRIVATE
    openvino::runtime openvino::frontend::onnx ${OpenCV_LIBS})

add_executable(lane_fps_ov src/lane_fps_test_ov.cpp)
target_include_directories(lane_fps_ov PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lane_fps_ov PRIVATE
    openvino::runtime openvino::frontend::onnx ${OpenCV_LIBS})


add_executable(lane_fps_ort src/lane_fps_test_ort.cpp)
target_include_directories(lane_fps_ort PRIVATE "${ORT_ROOT}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lane_fps_ort PRIVATE "${ORT_ROOT}/lib/libonnxruntime.so" ${OpenCV_LIBS})
set_target_properties(lane_fps_ort PROPERTIES BUILD_RPATH "${ORT_ROOT}/lib")
