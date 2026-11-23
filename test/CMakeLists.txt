# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# TODO: add tests for MM, MV, SPMV, SPMM, DenseNN, and SparseNN

add_executable( project_gemm_test
        gemm_test.cpp
        ${CMAKE_SOURCE_DIR}/src/gemm.cpp
        )
target_compile_options(project_gemm_test PRIVATE -mavx2 -mfma -march=native)
target_link_libraries( project_gemm_test
        PRIVATE
        GTest::gtest_main
        )
if(OPENMP)
    target_link_libraries(project_gemm_test PRIVATE OpenMP::OpenMP_CXX)
endif()
if(USE_MKL)
    target_link_libraries(project_gemm_test PRIVATE MKL::MKL)
    target_compile_definitions(project_gemm_test PRIVATE USE_MKL)
endif()

add_executable( project_gemv_test
        gemv_test.cpp
        ${CMAKE_SOURCE_DIR}/src/gemv.cpp
        )
target_compile_options(project_gemv_test PRIVATE -mavx2 -mfma -march=native)
target_link_libraries( project_gemv_test
        PRIVATE
        GTest::gtest_main
        )
if(OPENMP)
    target_link_libraries(project_gemv_test PRIVATE OpenMP::OpenMP_CXX)
endif()
if(USE_MKL)
    target_link_libraries(project_gemv_test PRIVATE MKL::MKL)
    target_compile_definitions(project_gemv_test PRIVATE USE_MKL)
endif()

add_executable( project_spmv_test
        spmv_test.cpp
        ${CMAKE_SOURCE_DIR}/src/spmv.cpp
        ${CMAKE_SOURCE_DIR}/src/gemv.cpp
        )
target_compile_options(project_spmv_test PRIVATE -mavx2 -mfma -march=native)
target_link_libraries( project_spmv_test
        PRIVATE
        GTest::gtest_main
        )
if(OPENMP)
    target_link_libraries(project_spmv_test PRIVATE OpenMP::OpenMP_CXX)
endif()
if(USE_MKL)
    target_link_libraries(project_spmv_test PRIVATE MKL::MKL)
    target_compile_definitions(project_spmv_test PRIVATE USE_MKL)
endif()

add_executable( project_spmm_test
        spmm_test.cpp
        ${CMAKE_SOURCE_DIR}/src/spmm.cpp
        ${CMAKE_SOURCE_DIR}/src/gemm.cpp
        )
target_compile_options(project_spmm_test PRIVATE -mavx2 -mfma -march=native)
target_link_libraries( project_spmm_test
        PRIVATE
        GTest::gtest_main
        )
if(OPENMP)
    target_link_libraries(project_spmm_test PRIVATE OpenMP::OpenMP_CXX)
endif()
if(USE_MKL)
    target_link_libraries(project_spmm_test PRIVATE MKL::MKL)
    target_compile_definitions(project_spmm_test PRIVATE USE_MKL)
endif()

if(CMAKE_CUDA_COMPILER)
    add_executable( project_gemm_gpu_test
            gemm_test.cu
            ${CMAKE_SOURCE_DIR}/src/kernels.cu
            )
    target_link_libraries( project_gemm_gpu_test
            PRIVATE
            GTest::gtest_main
            CUDA::cublas
            )

    add_executable( project_gemv_gpu_test
            gemv_test.cu
            ${CMAKE_SOURCE_DIR}/src/kernels.cu
            )
    target_link_libraries( project_gemv_gpu_test
            PRIVATE
            GTest::gtest_main
            CUDA::cublas
            )

    add_executable( project_spmv_gpu_test
            spmv_test.cu
            ${CMAKE_SOURCE_DIR}/src/kernels.cu
            )
    target_link_libraries( project_spmv_gpu_test
            PRIVATE
            GTest::gtest_main
            CUDA::cusparse
            )

    add_executable( project_spmm_gpu_test
            spmm_test.cu
            ${CMAKE_SOURCE_DIR}/src/kernels.cu
            )
    target_link_libraries( project_spmm_gpu_test
            PRIVATE
            GTest::gtest_main
            CUDA::cusparse
            )
endif()

include(GoogleTest)
gtest_discover_tests(project_gemm_test)
gtest_discover_tests(project_gemv_test)
gtest_discover_tests(project_spmv_test)
gtest_discover_tests(project_spmm_test)
