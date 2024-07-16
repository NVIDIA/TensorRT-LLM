find_package(GTest REQUIRED)

add_executable(tests tests.cpp)

target_link_libraries(tests
 PRIVATE
  GTest::GTest
  GTest::gtest_main
  tiktoken)

add_test(gtest tests)
FILE(COPY ../o200k_base.tiktoken ../cl100k_base.tiktoken ../p50k_base.tiktoken ../r50k_base.tiktoken DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/tokenizers")
