cmake_minimum_required(VERSION 3.16)
project(vectordb)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

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

# Google Test Setup
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP ON
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Test executable (unit tests)
add_executable(vectordb_test
    tests/test_vectordb.cpp
    tests/test_synthetic.cpp
    tests/test_benchmark.cpp
)

target_link_libraries(vectordb_test PRIVATE vectordb_lib gtest_main)
target_include_directories(vectordb_test PRIVATE include external)

add_test(NAME vectordb_tests COMMAND vectordb_test)

# CLI Integration Tests
add_executable(vectordb_cli_test
    tests/test_cli_integration.cpp
)

target_link_libraries(vectordb_cli_test PRIVATE gtest_main)
target_include_directories(vectordb_cli_test PRIVATE include external)

add_test(NAME vectordb_cli_tests COMMAND vectordb_cli_test WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})