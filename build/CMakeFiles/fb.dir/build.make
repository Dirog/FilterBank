# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dirog/Documents/FilterBankProject

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dirog/Documents/FilterBankProject/build

# Include any dependencies generated for this target.
include CMakeFiles/fb.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fb.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fb.dir/flags.make

CMakeFiles/fb.dir/source/main.cpp.o: CMakeFiles/fb.dir/flags.make
CMakeFiles/fb.dir/source/main.cpp.o: ../source/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dirog/Documents/FilterBankProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fb.dir/source/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fb.dir/source/main.cpp.o -c /home/dirog/Documents/FilterBankProject/source/main.cpp

CMakeFiles/fb.dir/source/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fb.dir/source/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dirog/Documents/FilterBankProject/source/main.cpp > CMakeFiles/fb.dir/source/main.cpp.i

CMakeFiles/fb.dir/source/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fb.dir/source/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dirog/Documents/FilterBankProject/source/main.cpp -o CMakeFiles/fb.dir/source/main.cpp.s

CMakeFiles/fb.dir/source/main.cpp.o.requires:

.PHONY : CMakeFiles/fb.dir/source/main.cpp.o.requires

CMakeFiles/fb.dir/source/main.cpp.o.provides: CMakeFiles/fb.dir/source/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/fb.dir/build.make CMakeFiles/fb.dir/source/main.cpp.o.provides.build
.PHONY : CMakeFiles/fb.dir/source/main.cpp.o.provides

CMakeFiles/fb.dir/source/main.cpp.o.provides.build: CMakeFiles/fb.dir/source/main.cpp.o


CMakeFiles/fb.dir/source/filterbank.cpp.o: CMakeFiles/fb.dir/flags.make
CMakeFiles/fb.dir/source/filterbank.cpp.o: ../source/filterbank.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dirog/Documents/FilterBankProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/fb.dir/source/filterbank.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fb.dir/source/filterbank.cpp.o -c /home/dirog/Documents/FilterBankProject/source/filterbank.cpp

CMakeFiles/fb.dir/source/filterbank.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fb.dir/source/filterbank.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dirog/Documents/FilterBankProject/source/filterbank.cpp > CMakeFiles/fb.dir/source/filterbank.cpp.i

CMakeFiles/fb.dir/source/filterbank.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fb.dir/source/filterbank.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dirog/Documents/FilterBankProject/source/filterbank.cpp -o CMakeFiles/fb.dir/source/filterbank.cpp.s

CMakeFiles/fb.dir/source/filterbank.cpp.o.requires:

.PHONY : CMakeFiles/fb.dir/source/filterbank.cpp.o.requires

CMakeFiles/fb.dir/source/filterbank.cpp.o.provides: CMakeFiles/fb.dir/source/filterbank.cpp.o.requires
	$(MAKE) -f CMakeFiles/fb.dir/build.make CMakeFiles/fb.dir/source/filterbank.cpp.o.provides.build
.PHONY : CMakeFiles/fb.dir/source/filterbank.cpp.o.provides

CMakeFiles/fb.dir/source/filterbank.cpp.o.provides.build: CMakeFiles/fb.dir/source/filterbank.cpp.o


# Object files for target fb
fb_OBJECTS = \
"CMakeFiles/fb.dir/source/main.cpp.o" \
"CMakeFiles/fb.dir/source/filterbank.cpp.o"

# External object files for target fb
fb_EXTERNAL_OBJECTS =

fb: CMakeFiles/fb.dir/source/main.cpp.o
fb: CMakeFiles/fb.dir/source/filterbank.cpp.o
fb: CMakeFiles/fb.dir/build.make
fb: CMakeFiles/fb.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dirog/Documents/FilterBankProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable fb"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fb.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fb.dir/build: fb

.PHONY : CMakeFiles/fb.dir/build

CMakeFiles/fb.dir/requires: CMakeFiles/fb.dir/source/main.cpp.o.requires
CMakeFiles/fb.dir/requires: CMakeFiles/fb.dir/source/filterbank.cpp.o.requires

.PHONY : CMakeFiles/fb.dir/requires

CMakeFiles/fb.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fb.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fb.dir/clean

CMakeFiles/fb.dir/depend:
	cd /home/dirog/Documents/FilterBankProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dirog/Documents/FilterBankProject /home/dirog/Documents/FilterBankProject /home/dirog/Documents/FilterBankProject/build /home/dirog/Documents/FilterBankProject/build /home/dirog/Documents/FilterBankProject/build/CMakeFiles/fb.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fb.dir/depend

