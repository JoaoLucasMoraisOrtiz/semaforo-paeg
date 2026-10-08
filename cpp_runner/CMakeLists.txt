cmake_minimum_required(VERSION 3.10)
project(rtdetr_runner)

set(CMAKE_CXX_STANDARD 14)

# Procura a biblioteca OpenCV no sistema
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Adiciona o executável
add_executable(rtdetr_runner main.cpp)

# Linka com o OpenCV
target_link_libraries(rtdetr_runner ${OpenCV_LIBS})
