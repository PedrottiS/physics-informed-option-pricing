cmake_minimum_required(VERSION 3.16)
project(pinn_bench_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(bs_fd src/main.cpp)
target_include_directories(bs_fd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bs_fd PRIVATE Eigen3::Eigen)
target_compile_options(bs_fd PRIVATE -O2 -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(bs_fd_test src/test.cpp)
target_include_directories(bs_fd_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bs_fd_test PRIVATE Eigen3::Eigen)
target_compile_options(bs_fd_test PRIVATE -O2 -Wall -Wextra -Wpedantic)
add_test(NAME unit_tests COMMAND bs_fd_test)