# 最低 CMake 版本
cmake_minimum_required(VERSION 3.10)

# 项目名
project(OnnxDemo)

# ONNX Runtime（Linux 版）位置
# $ENV{HOME} 就是你的家目录（比如 /home/jrong），换机器只改这一行

set(ORT_ROOT "$ENV{HOME}/onnxruntime-linux-x64-1.31.0")

# 源文件（想换程序就改这里）
#set(MAIN_SRC image_demo2.cpp)
#set(MAIN_SRC image_demo2_bug.cpp)
set(MAIN_SRC fps_test.cpp)

add_executable(app ${MAIN_SRC})


# OpenCV（让 CMake 自动找到它装在哪）
find_package(OpenCV REQUIRED)

# 头文件路径
target_include_directories(app PRIVATE "${ORT_ROOT}/include" ${OpenCV_INCLUDE_DIRS})

# 链接库（Linux 上是 .so，不是 Windows 的 .lib）
target_link_libraries(app PRIVATE "${ORT_ROOT}/lib/libonnxruntime.so" ${OpenCV_LIBS})

# 让程序运行时也能自己找到 libonnxruntime.so
# （不然每次跑都要先设 LD_LIBRARY_PATH，很烦）
set_target_properties(app PROPERTIES
    BUILD_RPATH   "${ORT_ROOT}/lib"
    INSTALL_RPATH "${ORT_ROOT}/lib")
