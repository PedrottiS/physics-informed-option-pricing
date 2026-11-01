cmake_minimum_required(VERSION 3.16)
project(pinn_bench_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# --- ONNX Runtime Setup ---
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")
if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime non trovato! Controlla che la cartella si chiami 'onnxruntime' e sia dentro cpp/")
endif()

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# --- Eseguibili ---
add_executable(bs_fd src/main.cpp)
target_include_directories(bs_fd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bs_fd PRIVATE Eigen3::Eigen onnxruntime)
target_compile_options(bs_fd PRIVATE -O2 -Wall -Wextra -Wpedantic)

# --- Test ---
enable_testing()
add_executable(bs_fd_test src/test.cpp)
target_include_directories(bs_fd_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bs_fd_test PRIVATE Eigen3::Eigen) # I test non usano ONNX
target_compile_options(bs_fd_test PRIVATE -O2 -Wall -Wextra -Wpedantic)
add_test(NAME unit_tests COMMAND bs_fd_test)