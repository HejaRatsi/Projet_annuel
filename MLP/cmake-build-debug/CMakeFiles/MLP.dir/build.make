# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\Users\hejar\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\201.6668.126\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = C:\Users\hejar\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\201.6668.126\bin\cmake\win\bin\cmake.exe -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\hejar\OneDrive\Bureau\ESGI\PA\Rendus\Rendu2Synthese\MLP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\hejar\OneDrive\Bureau\ESGI\PA\Rendus\Rendu2Synthese\MLP\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles\MLP.dir\depend.make

# Include the progress variables for this target.
include CMakeFiles\MLP.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\MLP.dir\flags.make

CMakeFiles\MLP.dir\main.cpp.obj: CMakeFiles\MLP.dir\flags.make
CMakeFiles\MLP.dir\main.cpp.obj: ..\main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\hejar\OneDrive\Bureau\ESGI\PA\Rendus\Rendu2Synthese\MLP\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MLP.dir/main.cpp.obj"
	C:\PROGRA~2\MICROS~3\2019\COMMUN~1\VC\Tools\MSVC\1425~1.286\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\MLP.dir\main.cpp.obj /FdCMakeFiles\MLP.dir\ /FS -c C:\Users\hejar\OneDrive\Bureau\ESGI\PA\Rendus\Rendu2Synthese\MLP\main.cpp
<<

CMakeFiles\MLP.dir\main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MLP.dir/main.cpp.i"
	C:\PROGRA~2\MICROS~3\2019\COMMUN~1\VC\Tools\MSVC\1425~1.286\bin\Hostx64\x64\cl.exe > CMakeFiles\MLP.dir\main.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\hejar\OneDrive\Bureau\ESGI\PA\Rendus\Rendu2Synthese\MLP\main.cpp
<<

CMakeFiles\MLP.dir\main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MLP.dir/main.cpp.s"
	C:\PROGRA~2\MICROS~3\2019\COMMUN~1\VC\Tools\MSVC\1425~1.286\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\MLP.dir\main.cpp.s /c C:\Users\hejar\OneDrive\Bureau\ESGI\PA\Rendus\Rendu2Synthese\MLP\main.cpp
<<

# Object files for target MLP
MLP_OBJECTS = \
"CMakeFiles\MLP.dir\main.cpp.obj"

# External object files for target MLP
MLP_EXTERNAL_OBJECTS =

MLP.exe: CMakeFiles\MLP.dir\main.cpp.obj
MLP.exe: CMakeFiles\MLP.dir\build.make
MLP.exe: CMakeFiles\MLP.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\hejar\OneDrive\Bureau\ESGI\PA\Rendus\Rendu2Synthese\MLP\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable MLP.exe"
	C:\Users\hejar\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\201.6668.126\bin\cmake\win\bin\cmake.exe -E vs_link_exe --intdir=CMakeFiles\MLP.dir --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\mt.exe --manifests  -- C:\PROGRA~2\MICROS~3\2019\COMMUN~1\VC\Tools\MSVC\1425~1.286\bin\Hostx64\x64\link.exe /nologo @CMakeFiles\MLP.dir\objects1.rsp @<<
 /out:MLP.exe /implib:MLP.lib /pdb:C:\Users\hejar\OneDrive\Bureau\ESGI\PA\Rendus\Rendu2Synthese\MLP\cmake-build-debug\MLP.pdb /version:0.0  /machine:x64 /debug /INCREMENTAL /subsystem:console  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib 
<<

# Rule to build all files generated by this target.
CMakeFiles\MLP.dir\build: MLP.exe

.PHONY : CMakeFiles\MLP.dir\build

CMakeFiles\MLP.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\MLP.dir\cmake_clean.cmake
.PHONY : CMakeFiles\MLP.dir\clean

CMakeFiles\MLP.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\hejar\OneDrive\Bureau\ESGI\PA\Rendus\Rendu2Synthese\MLP C:\Users\hejar\OneDrive\Bureau\ESGI\PA\Rendus\Rendu2Synthese\MLP C:\Users\hejar\OneDrive\Bureau\ESGI\PA\Rendus\Rendu2Synthese\MLP\cmake-build-debug C:\Users\hejar\OneDrive\Bureau\ESGI\PA\Rendus\Rendu2Synthese\MLP\cmake-build-debug C:\Users\hejar\OneDrive\Bureau\ESGI\PA\Rendus\Rendu2Synthese\MLP\cmake-build-debug\CMakeFiles\MLP.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\MLP.dir\depend

