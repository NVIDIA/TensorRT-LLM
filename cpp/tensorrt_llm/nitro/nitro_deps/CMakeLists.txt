cmake_minimum_required(VERSION 3.22)  # Required for FetchContent

project(MyProject)

include(ExternalProject)

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
	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    jsoncpp
    GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp
    GIT_TAG 1.9.5
    CMAKE_ARGS 
    	-DBUILD_SHARED_LIBS=OFF
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    c-ares
    GIT_REPOSITORY https://github.com/c-ares/c-ares
    GIT_TAG cares-1_26_0
    CMAKE_ARGS
    	-DCARES_SHARED=OFF
	-DCARES_STATIC=ON
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    drogon
    GIT_REPOSITORY https://github.com/drogonframework/drogon
    GIT_TAG v1.9.2
    CMAKE_ARGS
	-DCMAKE_BUILD_TYPE=release
	-DOPENSSL_USE_STATIC_LIBS=TRUE
	-DZLIB_USE_STATIC_LIBS=${ZLIB_USE_STATIC_LIBS}
	-DBUILD_ORM=OFF
	-DBUILD_YAML_CONFIG=OFF
	-DBUILD_EXAMPLES=OFF
	-DBUILD_CTL=OFF
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	-DBUILD_BROTLI=ON
	-DCMAKE_PREFIX_PATH=${THIRD_PARTY_INSTALL_PATH}
	# -DCMAKE_FIND_ROOT_PATH=${THIRD_PARTY_INSTALL_PATH} # To set the dir (that will be used to force the look for .a)
	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece
    GIT_TAG v0.2.0
    CMAKE_ARGS
    	-DSPM_ENABLE_SHARED=OFF
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

# Fix trantor cmakelists to link c-ares on Windows
if(WIN32)
    set(TRANTOR_CMAKE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../build_deps/nitro_deps/drogon-prefix/src/drogon/trantor/CMakeLists.txt)
    ExternalProject_Add_Step(drogon trantor_custom_target
        COMMAND ${CMAKE_COMMAND} -E echo add_definitions(-DCARES_STATICLIB) >> ${TRANTOR_CMAKE_FILE}
	DEPENDEES download
    )
endif()

include_directories(${THIRD_PARTY_INSTALL_PATH}/include)
link_directories(${THIRD_PARTY_INSTALL_PATH}/lib)
# Optionally link or add dependencies to your targets
add_dependencies(drogon c-ares jsoncpp brotli)

if(ZLIB_USE_STATIC_LIBS)
    add_dependencies(drogon zlib)
endif()
# target_link_libraries(<your-target> ...)
