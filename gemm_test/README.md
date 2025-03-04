# tf32_gemm_example Build and Usage Guide

This guide explains step by step how to set up, modify, and build the tf32_gemm_example project using the provided CMakeLists.txt. This project is built on CUTLASS and can be used by all users by setting an environment variable for the CUTLASS repository path.

---

## 1. Set Up Your Environment

1. **Set the `CUTLASS_PATH` Environment Variable:**
   - Point `CUTLASS_PATH` to the root directory of your CUTLASS repository.
   - For example, if your CUTLASS repository is in `~/workspace/DiDt-Power-Proxy/cutlass`, then run:
     ```bash
     export CUTLASS_PATH=~/workspace/DiDt-Power-Proxy/cutlass
     ```
   - You may add this line to your shell startup file (e.g., `.bashrc`) so it’s set automatically.

---

## 2. Prepare the Project Workspace

1. **Create a New Project Folder:**
   - Create a folder outside the CUTLASS repository for your example project.
   - For example:
     ```bash
     mkdir ~/workspace/gemm_test
     cd ~/workspace/gemm_test
     ```

2. **Copy the Example Source File:**
   - Copy the desired example from the CUTLASS repository into your project folder.
   - For instance, copy `ampere_tf32_tensorop_gemm.cu` from:
     ```
     ${CUTLASS_PATH}/examples/14_ampere_tf32_tensorop_gemm/
     ```
     to your project folder:
     ```bash
     cp ${CUTLASS_PATH}/examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm.cu .
     ```

3. **Create Your Custom CMakeLists.txt:**
   - In your project folder, create a file named `CMakeLists.txt` with the following content:

     ```cmake
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
     ```

   - **Note:** If you wish to use another example (e.g., a different .cu file), change the filename in both the `add_executable` and the custom command. Also, add any additional include directories required by that example.

---

## 3. Build the Project

1. **Create a Build Directory and Configure:**
   - From your project folder:
     ```bash
     mkdir build && cd build
     cmake ..
     ```
   - This command will process your CMakeLists.txt and set up the build system.

2. **Compile the Project:**
   - Run:
     ```bash
     make
     ```
   - The build process will compile the `tf32_gemm` executable and also generate the PTX file.

---

## 4. Running the Application and PTX Generation

1. **Run the Executable:**
   - The generated executable `tf32_gemm` accepts command-line parameters such as matrix sizes.
   - For example:
     ```bash
     ./tf32_gemm --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707
     ```

2. **Locate the PTX File:**
   - The PTX file is generated in the build directory as `ampere_tf32_tensorop_gemm.ptx`.
   - You can inspect this file or use it as needed.

---

## 5. Changing Examples or Adding More Includes

- **Switching to a Different Example:**
  - Copy the desired `.cu` file from the appropriate CUTLASS example folder into your project.
  - Update the `add_executable` line and the custom command in your CMakeLists.txt with the new filename.

- **Adding More Include Directories:**
  - If your example requires additional headers, add their paths inside the `include_directories()` call.
  - Use relative paths based on the environment variable `CUTLASS_PATH` to maintain portability.

---

## 6. Summary

- **Set the Environment:**  
  Make sure `CUTLASS_PATH` is set.

- **Prepare Your Workspace:**  
  Create a separate project folder and copy the desired example.

- **Customize CMakeLists.txt:**  
  Use the provided template, adjusting the filenames and include paths if needed.

- **Build and Run:**  
  Create a build directory, run `cmake ..` and `make`, then run your executable with the desired parameters.  
  The PTX file is generated automatically by the custom target.


