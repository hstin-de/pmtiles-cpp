cmake_minimum_required(VERSION 3.16)
project(pmtiles-cpp VERSION 0.1.0 LANGUAGES CXX)

option(PMTILES_BUILD_SHARED "Build shared library" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

# ---- Dependencies ----
find_package(ZLIB REQUIRED)

find_path(XXHASH_INCLUDE_DIR xxhash.h)
if(NOT XXHASH_INCLUDE_DIR)
  message(FATAL_ERROR "xxhash.h not found. On Arch: pacman -S xxhash")
endif()
find_library(XXHASH_LIBRARY NAMES xxhash)
if(NOT XXHASH_LIBRARY)
  message(FATAL_ERROR "libxxhash not found. On Arch: pacman -S xxhash")
endif()

# ---- Library ----
if(PMTILES_BUILD_SHARED)
  add_library(pmtiles SHARED src/pmtiles.cpp)
else()
  add_library(pmtiles STATIC src/pmtiles.cpp)
endif()

# Position-independent for both (safe on all platforms)
set_target_properties(pmtiles PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(pmtiles
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${XXHASH_INCLUDE_DIR}
)

target_link_libraries(pmtiles
  PUBLIC
    ZLIB::ZLIB
  PRIVATE
    ${XXHASH_LIBRARY}
)

# Export symbol on Windows
target_compile_definitions(pmtiles PRIVATE PMTILES_BUILDING_LIB=1)

# Namespace alias
add_library(pmtiles::pmtiles ALIAS pmtiles)

# ---- Install / Package ----
install(TARGETS pmtiles
  EXPORT pmtilesTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES include/pmtiles.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT pmtilesTargets
  FILE pmtilesTargets.cmake
  NAMESPACE pmtiles::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pmtiles
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/pmtilesConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/pmtilesConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pmtiles"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pmtilesConfig.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pmtiles"
)

message(STATUS "pmtiles-cpp ${PROJECT_VERSION}")
message(STATUS "xxhash include: ${XXHASH_INCLUDE_DIR}")
message(STATUS "xxhash lib:     ${XXHASH_LIBRARY}")
