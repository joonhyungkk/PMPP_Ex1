# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1

# Include any dependencies generated for this target.
include CMakeFiles/matmul.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/matmul.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matmul.dir/flags.make

CMakeFiles/matmul.dir/main.cc.o: CMakeFiles/matmul.dir/flags.make
CMakeFiles/matmul.dir/main.cc.o: main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/matmul.dir/main.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/matmul.dir/main.cc.o -c /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/main.cc

CMakeFiles/matmul.dir/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matmul.dir/main.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/main.cc > CMakeFiles/matmul.dir/main.cc.i

CMakeFiles/matmul.dir/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matmul.dir/main.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/main.cc -o CMakeFiles/matmul.dir/main.cc.s

CMakeFiles/matmul.dir/matmul.cu.o: CMakeFiles/matmul.dir/flags.make
CMakeFiles/matmul.dir/matmul.cu.o: matmul.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/matmul.dir/matmul.cu.o"
	/usr/local/cuda-11.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/matmul.cu -o CMakeFiles/matmul.dir/matmul.cu.o

CMakeFiles/matmul.dir/matmul.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/matmul.dir/matmul.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/matmul.dir/matmul.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/matmul.dir/matmul.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/matmul.dir/matrix.cu.o: CMakeFiles/matmul.dir/flags.make
CMakeFiles/matmul.dir/matrix.cu.o: matrix.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/matmul.dir/matrix.cu.o"
	/usr/local/cuda-11.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/matrix.cu -o CMakeFiles/matmul.dir/matrix.cu.o

CMakeFiles/matmul.dir/matrix.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/matmul.dir/matrix.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/matmul.dir/matrix.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/matmul.dir/matrix.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/matmul.dir/test.cc.o: CMakeFiles/matmul.dir/flags.make
CMakeFiles/matmul.dir/test.cc.o: test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/matmul.dir/test.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/matmul.dir/test.cc.o -c /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/test.cc

CMakeFiles/matmul.dir/test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matmul.dir/test.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/test.cc > CMakeFiles/matmul.dir/test.cc.i

CMakeFiles/matmul.dir/test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matmul.dir/test.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/test.cc -o CMakeFiles/matmul.dir/test.cc.s

CMakeFiles/matmul.dir/mul_cpu.cc.o: CMakeFiles/matmul.dir/flags.make
CMakeFiles/matmul.dir/mul_cpu.cc.o: mul_cpu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/matmul.dir/mul_cpu.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/matmul.dir/mul_cpu.cc.o -c /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/mul_cpu.cc

CMakeFiles/matmul.dir/mul_cpu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matmul.dir/mul_cpu.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/mul_cpu.cc > CMakeFiles/matmul.dir/mul_cpu.cc.i

CMakeFiles/matmul.dir/mul_cpu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matmul.dir/mul_cpu.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/mul_cpu.cc -o CMakeFiles/matmul.dir/mul_cpu.cc.s

CMakeFiles/matmul.dir/mul_gpu.cu.o: CMakeFiles/matmul.dir/flags.make
CMakeFiles/matmul.dir/mul_gpu.cu.o: mul_gpu.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CUDA object CMakeFiles/matmul.dir/mul_gpu.cu.o"
	/usr/local/cuda-11.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/mul_gpu.cu -o CMakeFiles/matmul.dir/mul_gpu.cu.o

CMakeFiles/matmul.dir/mul_gpu.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/matmul.dir/mul_gpu.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/matmul.dir/mul_gpu.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/matmul.dir/mul_gpu.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target matmul
matmul_OBJECTS = \
"CMakeFiles/matmul.dir/main.cc.o" \
"CMakeFiles/matmul.dir/matmul.cu.o" \
"CMakeFiles/matmul.dir/matrix.cu.o" \
"CMakeFiles/matmul.dir/test.cc.o" \
"CMakeFiles/matmul.dir/mul_cpu.cc.o" \
"CMakeFiles/matmul.dir/mul_gpu.cu.o"

# External object files for target matmul
matmul_EXTERNAL_OBJECTS =

matmul: CMakeFiles/matmul.dir/main.cc.o
matmul: CMakeFiles/matmul.dir/matmul.cu.o
matmul: CMakeFiles/matmul.dir/matrix.cu.o
matmul: CMakeFiles/matmul.dir/test.cc.o
matmul: CMakeFiles/matmul.dir/mul_cpu.cc.o
matmul: CMakeFiles/matmul.dir/mul_gpu.cu.o
matmul: CMakeFiles/matmul.dir/build.make
matmul: CMakeFiles/matmul.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable matmul"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matmul.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matmul.dir/build: matmul

.PHONY : CMakeFiles/matmul.dir/build

CMakeFiles/matmul.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/matmul.dir/cmake_clean.cmake
.PHONY : CMakeFiles/matmul.dir/clean

CMakeFiles/matmul.dir/depend:
	cd /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1 /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1 /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1 /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1 /gris/gris-f/homelv/jkim/PMPP_Ex1/ex1/CMakeFiles/matmul.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/matmul.dir/depend

