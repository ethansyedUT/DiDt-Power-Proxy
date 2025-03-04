cmake_minimum_required(VERSION 3.18)
project(tf32_gemm_example LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set the CUDA architecture for Ampere (e.g., sm_80)  
#### Need change if using a different Arch ####
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")

# Check for CUTLASS_PATH environment variable.
if(NOT DEFINED ENV{CUTLASS_PATH})
  message(FATAL_ERROR "Please set the CUTLASS_PATH environment variable to your CUTLASS repository location.")
endif()
set(CUTLASS_PATH $ENV{CUTLASS_PATH})

# Include CUTLASS headers and additional utility headers  
#### If error pop out due to incude, you need to add or change the include ####
include_directories(
  ${CUTLASS_PATH}/include
  ${CUTLASS_PATH}/tools/util/include
  ${CUTLASS_PATH}/examples/common
)

# Define the executable target for your example
#### change the file name if needed, first argument is executable, second is the main.cu ####
add_executable(tf32_gemm ampere_tf32_tensorop_gemm.cu)

# (Remove linking against CUTLASS since it is header-only in our case)
# target_link_libraries(tf32_gemm PRIVATE CUTLASS)

# --- Custom Target for PTX Generation ---
#### make sure the file name is consistent ####
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ampere_tf32_tensorop_gemm.ptx
  COMMAND ${CMAKE_CUDA_COMPILER} -ptx
          ${CMAKE_CURRENT_SOURCE_DIR}/ampere_tf32_tensorop_gemm.cu
          -I${CUTLASS_PATH}/include
          -I${CUTLASS_PATH}/tools/util/include
          -I${CUTLASS_PATH}/examples/common
          -arch=sm_80
          -o ${CMAKE_CURRENT_BINARY_DIR}/ampere_tf32_tensorop_gemm.ptx
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ampere_tf32_tensorop_gemm.cu
  COMMENT "Generating PTX file..."
)

# Create a custom target that depends on the generated PTX file.
#### make sure the file name is consistent ####
add_custom_target(generate_ptx ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ampere_tf32_tensorop_gemm.ptx)