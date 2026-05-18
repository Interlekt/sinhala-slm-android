cmake_minimum_required(VERSION 3.22.1)
project(slmengine)

# Disable things we don't need — speeds up build
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

add_library(llama_jni SHARED llama_jni.cpp)

target_link_libraries(llama_jni
        llama
        android
        log)
