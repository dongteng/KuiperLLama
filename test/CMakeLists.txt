include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest) #创建一个变量 link_ext_lib，里面保存了后续要链接的库名。
aux_source_directory(../test DIR_TEST)  # 自动查找 ../test 文件夹下所有 .cpp 文件，并保存到变量 DIR_TEST。这组文件会编译进项目里。

aux_source_directory(../test/test_cu DIR_TEST_CU)
aux_source_directory(../test/test_op DIR_TEST_OP)
aux_source_directory(../test/test_model DIR_TEST_MODEL)
aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)
aux_source_directory(../test/optimized DIR_TEST_OPTIMIZED)

add_executable(test_llm ${DIR_TEST} ${DIR_TEST_CU} ${DIR_TEST_OP} ${DIR_TEST_OPTIMIZED} ${DIR_TEST_TENSOR} ${DIR_TEST_MODEL}
        test_tensor/test_math.cpp) #把前面收集的 .cpp 文件统统编译为可执行文件 test_llm

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G") #这行被注释了。如果开启，会让 CUDA 以调试模式编译（-g 是生成调试符号，-G 是禁用所有优化）。
target_link_libraries(test_llm ${link_ext_lib}) # 把 glog 和 gtest 链接到 test_llm 可执行文件上。
target_include_directories(test_llm PUBLIC ${glog_INCLUDE_DIR}) #把 glog 和 gtest 的头文件路径加入编译器的 include 搜索路径中。
target_include_directories(test_llm PUBLIC ${GTest_INCLUDE_DIR})

target_include_directories(test_llm PUBLIC ../kuiper/include)  #添加你项目中自定义的头文件路径，可能定义了基础 Tensor、Model 类等。
target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/lib)
if (LLAMA3_SUPPORT OR QWEN2_SUPPORT)
    message(STATUS "LINK LLAMA3 SUPPORT")
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
    target_link_libraries(llama absl::base re2::re2 nlohmann_json::nlohmann_json)
endif ()
target_link_libraries(test_llm llama)

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)