# Generated by CMake

if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" LESS 2.8)
   message(FATAL_ERROR "CMake >= 2.8.0 required")
endif()
if(CMAKE_VERSION VERSION_LESS "2.8.3")
   message(FATAL_ERROR "CMake >= 2.8.3 required")
endif()
cmake_policy(PUSH)
cmake_policy(VERSION 2.8.3...3.24)
#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Protect against multiple inclusion, which would fail when already imported targets are added once more.
set(_cmake_targets_defined "")
set(_cmake_targets_not_defined "")
set(_cmake_expected_targets "")
foreach(_cmake_expected_target IN ITEMS hdf5::hdf5-static hdf5::hdf5-shared hdf5::mirror_server hdf5::mirror_server_stop hdf5::hdf5_tools-static hdf5::hdf5_tools-shared hdf5::h5diff hdf5::h5diff-shared hdf5::h5ls hdf5::h5ls-shared hdf5::h5debug hdf5::h5repart hdf5::h5mkgrp hdf5::h5clear hdf5::h5delete hdf5::h5debug-shared hdf5::h5repart-shared hdf5::h5mkgrp-shared hdf5::h5clear-shared hdf5::h5delete-shared hdf5::h5import hdf5::h5import-shared hdf5::h5repack hdf5::h5repack-shared hdf5::h5jam hdf5::h5unjam hdf5::h5jam-shared hdf5::h5unjam-shared hdf5::h5copy hdf5::h5copy-shared hdf5::h5stat hdf5::h5stat-shared hdf5::h5dump hdf5::h5dump-shared hdf5::h5format_convert hdf5::h5format_convert-shared hdf5::h5perf_serial hdf5::hdf5_hl-static hdf5::hdf5_hl-shared hdf5::h5watch hdf5::h5watch-shared hdf5::hdf5_cpp-static hdf5::hdf5_cpp-shared hdf5::hdf5_hl_cpp-static hdf5::hdf5_hl_cpp-shared)
  list(APPEND _cmake_expected_targets "${_cmake_expected_target}")
  if(TARGET "${_cmake_expected_target}")
    list(APPEND _cmake_targets_defined "${_cmake_expected_target}")
  else()
    list(APPEND _cmake_targets_not_defined "${_cmake_expected_target}")
  endif()
endforeach()
unset(_cmake_expected_target)
if(_cmake_targets_defined STREQUAL _cmake_expected_targets)
  unset(_cmake_targets_defined)
  unset(_cmake_targets_not_defined)
  unset(_cmake_expected_targets)
  unset(CMAKE_IMPORT_FILE_VERSION)
  cmake_policy(POP)
  return()
endif()
if(NOT _cmake_targets_defined STREQUAL "")
  string(REPLACE ";" ", " _cmake_targets_defined_text "${_cmake_targets_defined}")
  string(REPLACE ";" ", " _cmake_targets_not_defined_text "${_cmake_targets_not_defined}")
  message(FATAL_ERROR "Some (but not all) targets in this export set were already defined.\nTargets Defined: ${_cmake_targets_defined_text}\nTargets not yet defined: ${_cmake_targets_not_defined_text}\n")
endif()
unset(_cmake_targets_defined)
unset(_cmake_targets_not_defined)
unset(_cmake_expected_targets)


# Compute the installation prefix relative to this file.
get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
if(_IMPORT_PREFIX STREQUAL "/")
  set(_IMPORT_PREFIX "")
endif()

# Create imported target hdf5::hdf5-static
add_library(hdf5::hdf5-static STATIC IMPORTED)

set_target_properties(hdf5::hdf5-static PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include;${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "\$<LINK_ONLY:shlwapi>;\$<\$<NOT:\$<PLATFORM_ID:Windows>>:>;\$<\$<BOOL:OFF>:MPI::MPI_C>"
)

# Create imported target hdf5::hdf5-shared
add_library(hdf5::hdf5-shared SHARED IMPORTED)

set_target_properties(hdf5::hdf5-shared PROPERTIES
  INTERFACE_COMPILE_DEFINITIONS "H5_BUILT_AS_DYNAMIC_LIB"
  INTERFACE_INCLUDE_DIRECTORIES "\$<\$<BOOL:OFF>:>;${_IMPORT_PREFIX}/include;${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "\$<\$<NOT:\$<PLATFORM_ID:Windows>>:>;\$<\$<BOOL:OFF>:MPI::MPI_C>"
)

# Create imported target hdf5::mirror_server
add_executable(hdf5::mirror_server IMPORTED)

# Create imported target hdf5::mirror_server_stop
add_executable(hdf5::mirror_server_stop IMPORTED)

# Create imported target hdf5::hdf5_tools-static
add_library(hdf5::hdf5_tools-static STATIC IMPORTED)

set_target_properties(hdf5::hdf5_tools-static PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include;${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "hdf5::hdf5-static;\$<LINK_ONLY:\$<\$<BOOL:OFF>:MPI::MPI_C>>"
)

# Create imported target hdf5::hdf5_tools-shared
add_library(hdf5::hdf5_tools-shared SHARED IMPORTED)

set_target_properties(hdf5::hdf5_tools-shared PROPERTIES
  INTERFACE_COMPILE_DEFINITIONS "H5_BUILT_AS_DYNAMIC_LIB"
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include;${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "hdf5::hdf5-shared"
)

# Create imported target hdf5::h5diff
add_executable(hdf5::h5diff IMPORTED)

# Create imported target hdf5::h5diff-shared
add_executable(hdf5::h5diff-shared IMPORTED)

# Create imported target hdf5::h5ls
add_executable(hdf5::h5ls IMPORTED)

# Create imported target hdf5::h5ls-shared
add_executable(hdf5::h5ls-shared IMPORTED)

# Create imported target hdf5::h5debug
add_executable(hdf5::h5debug IMPORTED)

# Create imported target hdf5::h5repart
add_executable(hdf5::h5repart IMPORTED)

# Create imported target hdf5::h5mkgrp
add_executable(hdf5::h5mkgrp IMPORTED)

# Create imported target hdf5::h5clear
add_executable(hdf5::h5clear IMPORTED)

# Create imported target hdf5::h5delete
add_executable(hdf5::h5delete IMPORTED)

# Create imported target hdf5::h5debug-shared
add_executable(hdf5::h5debug-shared IMPORTED)

# Create imported target hdf5::h5repart-shared
add_executable(hdf5::h5repart-shared IMPORTED)

# Create imported target hdf5::h5mkgrp-shared
add_executable(hdf5::h5mkgrp-shared IMPORTED)

# Create imported target hdf5::h5clear-shared
add_executable(hdf5::h5clear-shared IMPORTED)

# Create imported target hdf5::h5delete-shared
add_executable(hdf5::h5delete-shared IMPORTED)

# Create imported target hdf5::h5import
add_executable(hdf5::h5import IMPORTED)

# Create imported target hdf5::h5import-shared
add_executable(hdf5::h5import-shared IMPORTED)

# Create imported target hdf5::h5repack
add_executable(hdf5::h5repack IMPORTED)

# Create imported target hdf5::h5repack-shared
add_executable(hdf5::h5repack-shared IMPORTED)

# Create imported target hdf5::h5jam
add_executable(hdf5::h5jam IMPORTED)

# Create imported target hdf5::h5unjam
add_executable(hdf5::h5unjam IMPORTED)

# Create imported target hdf5::h5jam-shared
add_executable(hdf5::h5jam-shared IMPORTED)

# Create imported target hdf5::h5unjam-shared
add_executable(hdf5::h5unjam-shared IMPORTED)

# Create imported target hdf5::h5copy
add_executable(hdf5::h5copy IMPORTED)

# Create imported target hdf5::h5copy-shared
add_executable(hdf5::h5copy-shared IMPORTED)

# Create imported target hdf5::h5stat
add_executable(hdf5::h5stat IMPORTED)

# Create imported target hdf5::h5stat-shared
add_executable(hdf5::h5stat-shared IMPORTED)

# Create imported target hdf5::h5dump
add_executable(hdf5::h5dump IMPORTED)

# Create imported target hdf5::h5dump-shared
add_executable(hdf5::h5dump-shared IMPORTED)

# Create imported target hdf5::h5format_convert
add_executable(hdf5::h5format_convert IMPORTED)

# Create imported target hdf5::h5format_convert-shared
add_executable(hdf5::h5format_convert-shared IMPORTED)

# Create imported target hdf5::h5perf_serial
add_executable(hdf5::h5perf_serial IMPORTED)

# Create imported target hdf5::hdf5_hl-static
add_library(hdf5::hdf5_hl-static STATIC IMPORTED)

set_target_properties(hdf5::hdf5_hl-static PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include;${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "hdf5::hdf5-static"
)

# Create imported target hdf5::hdf5_hl-shared
add_library(hdf5::hdf5_hl-shared SHARED IMPORTED)

set_target_properties(hdf5::hdf5_hl-shared PROPERTIES
  INTERFACE_COMPILE_DEFINITIONS "H5_BUILT_AS_DYNAMIC_LIB"
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include;${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "hdf5::hdf5-shared"
)

# Create imported target hdf5::h5watch
add_executable(hdf5::h5watch IMPORTED)

# Create imported target hdf5::h5watch-shared
add_executable(hdf5::h5watch-shared IMPORTED)

# Create imported target hdf5::hdf5_cpp-static
add_library(hdf5::hdf5_cpp-static STATIC IMPORTED)

set_target_properties(hdf5::hdf5_cpp-static PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include;${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "hdf5::hdf5-static"
)

# Create imported target hdf5::hdf5_cpp-shared
add_library(hdf5::hdf5_cpp-shared SHARED IMPORTED)

set_target_properties(hdf5::hdf5_cpp-shared PROPERTIES
  INTERFACE_COMPILE_DEFINITIONS "H5_BUILT_AS_DYNAMIC_LIB"
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include;${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "hdf5::hdf5-shared"
)

# Create imported target hdf5::hdf5_hl_cpp-static
add_library(hdf5::hdf5_hl_cpp-static STATIC IMPORTED)

set_target_properties(hdf5::hdf5_hl_cpp-static PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include;${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "hdf5::hdf5_hl-static;hdf5::hdf5-static"
)

# Create imported target hdf5::hdf5_hl_cpp-shared
add_library(hdf5::hdf5_hl_cpp-shared SHARED IMPORTED)

set_target_properties(hdf5::hdf5_hl_cpp-shared PROPERTIES
  INTERFACE_COMPILE_DEFINITIONS "H5_BUILT_AS_DYNAMIC_LIB"
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include;${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "hdf5::hdf5_hl-shared;hdf5::hdf5-shared"
)

if(CMAKE_VERSION VERSION_LESS 2.8.12)
  message(FATAL_ERROR "This file relies on consumers using CMake 2.8.12 or greater.")
endif()

# Load information for each installed configuration.
file(GLOB _cmake_config_files "${CMAKE_CURRENT_LIST_DIR}/hdf5-targets-*.cmake")
foreach(_cmake_config_file IN LISTS _cmake_config_files)
  include("${_cmake_config_file}")
endforeach()
unset(_cmake_config_file)
unset(_cmake_config_files)

# Cleanup temporary variables.
set(_IMPORT_PREFIX)

# Loop over all imported files and verify that they actually exist
foreach(_cmake_target IN LISTS _cmake_import_check_targets)
  foreach(_cmake_file IN LISTS "_cmake_import_check_files_for_${_cmake_target}")
    if(NOT EXISTS "${_cmake_file}")
      message(FATAL_ERROR "The imported target \"${_cmake_target}\" references the file
   \"${_cmake_file}\"
but this file does not exist.  Possible reasons include:
* The file was deleted, renamed, or moved to another location.
* An install or uninstall procedure did not complete successfully.
* The installation package was faulty and contained
   \"${CMAKE_CURRENT_LIST_FILE}\"
but not all the files it references.
")
    endif()
  endforeach()
  unset(_cmake_file)
  unset("_cmake_import_check_files_for_${_cmake_target}")
endforeach()
unset(_cmake_target)
unset(_cmake_import_check_targets)

# This file does not depend on other imported targets which have
# been exported from the same project but in a separate export set.

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
cmake_policy(POP)
