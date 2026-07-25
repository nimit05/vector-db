cmake_minimum_required(VERSION 3.16)
project(vectordb)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vectordb_lib
    src/collection.cpp
    src/similarity.cpp
    src/vectorRecord.cpp
)

target_include_directories(vectordb_lib PUBLIC include external)

add_executable(vectordb
    src/main.cpp
)

target_link_libraries(vectordb PRIVATE vectordb_lib)