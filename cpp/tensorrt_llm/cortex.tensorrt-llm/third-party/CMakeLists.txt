cmake_minimum_required(VERSION 3.22)  # Required for FetchContent

project(MyProject)

include(ExternalProject)

if(UNIX AND NOT APPLE)
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")
  add_compile_options(-fPIC)
endif()

# Define variables
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(THIRD_PARTY_INSTALL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../build_deps/_install)
#if(NOT THIRD_PARTY_INSTALL_PATH )
#  message(FATAL_ERROR "TRITON_THIRD_PARTY_INSTALL_PREFIX must be set")
#endif() # TRITON_THIRD_PARTY_INSTALL_PREFIX
# To force the find_package to look for .a inside self installed version
#set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
#set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
#set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
#
# Add the external project
set(ZLIB_USE_STATIC_LIBS OFF)
find_package(ZLIB)
if(NOT ZLIB_FOUND)
    set(ZLIB_USE_STATIC_LIBS ON)
    ExternalProject_Add(
        zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG v1.2.11
        CMAKE_ARGS
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
            -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
    )
endif()

ExternalProject_Add(
    brotli
    GIT_REPOSITORY https://github.com/google/brotli
    GIT_TAG v1.1.0
    CMAKE_ARGS 
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=OFF
        -DSHARE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/share
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    jsoncpp
    GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp
    GIT_TAG 1.9.5
    CMAKE_ARGS 
    	-DBUILD_SHARED_LIBS=OFF
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    c-ares
    GIT_REPOSITORY https://github.com/c-ares/c-ares
    GIT_TAG cares-1_26_0
    CMAKE_ARGS
    	-DCARES_SHARED=OFF
	    -DCARES_STATIC=ON
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    trantor
    GIT_REPOSITORY https://github.com/an-tao/trantor
    GIT_TAG v1.5.17
    CMAKE_ARGS 
    	-DBUILD_SHARED_LIBS=OFF
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_PREFIX_PATH=${THIRD_PARTY_INSTALL_PATH}
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece
    GIT_TAG v0.2.0
    CMAKE_ARGS
    	-DSPM_ENABLE_SHARED=OFF
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

include_directories(${THIRD_PARTY_INSTALL_PATH}/include)
link_directories(${THIRD_PARTY_INSTALL_PATH}/lib)
# Optionally link or add dependencies to your targets
add_dependencies(trantor c-ares jsoncpp brotli)

# target_link_libraries(<your-target> ...)
