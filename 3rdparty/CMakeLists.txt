include(ExternalProject)
include(FetchContent)

if(DEFINED ENV{GITHUB_MIRROR} AND NOT "$ENV{GITHUB_MIRROR}" STREQUAL "")
  set(github_base_url "$ENV{GITHUB_MIRROR}")
else()
  set(github_base_url "https://github.com")
endif()

# cmake-format: off
# ── FetchContent cache acceleration ──────────────────────────────────────
# Bare reference repos under $TRTLLM_FETCHCONTENT_CACHE (default:
# <3rdparty>/.cache_3rdparty) accelerate git clone via --reference.
# Set TRTLLM_FETCHCONTENT_CACHE="" to disable.
#
# Two parts:
#   1. fetch_cache_wrapper.py (GIT_EXECUTABLE override) injects
#      --reference into every clone spawned by FetchContent.
#   2. The FetchContent_MakeAvailable override below runs fetch_cache.py
#      update --src <dep_SOURCE_DIR> after each populate.  It runs in
#      the outer cmake scope on purpose: FetchContent's subbuild output
#      is swallowed by FETCHCONTENT_QUIET=ON, the outer scope's isn't.
# ─────────────────────────────────────────────────────────────────────────
# cmake-format: on
set(_FC_WRAPPER "${CMAKE_CURRENT_SOURCE_DIR}/fetch_cache_wrapper.py")

if(NOT WIN32 AND EXISTS "${_FC_WRAPPER}")

  if(DEFINED ENV{TRTLLM_FETCHCONTENT_CACHE})
    set(_FC_CACHE_DIR "$ENV{TRTLLM_FETCHCONTENT_CACHE}")
  else()
    set(_FC_CACHE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.cache_3rdparty")
  endif()

  if(NOT "${_FC_CACHE_DIR}" STREQUAL "")
    # Keep the real git in a separate cache var: GIT_EXECUTABLE gets overwritten
    # below, and find_package(Git) honors that cached value, so reusing the same
    # var would self-recurse on re-configure.
    find_program(_FC_REAL_GIT NAMES git REQUIRED)
    find_program(_FC_PYTHON NAMES python3 python REQUIRED)
    set(ENV{_TRTLLM_REAL_GIT} "${_FC_REAL_GIT}")
    set(ENV{TRTLLM_FETCHCONTENT_CACHE} "${_FC_CACHE_DIR}")
    set(GIT_EXECUTABLE
        "${_FC_WRAPPER}"
        CACHE FILEPATH "Git wrapper with reference cache acceleration" FORCE)
    # CACHE INTERNAL so sibling subdirs (cpp/tests, cpp/kernels/xqa,
    # cpp/tensorrt_llm/deep_ep, ...) see these when they call
    # FetchContent_MakeAvailable after this add_subdirectory returns.
    set(_FC_CACHE_DIR
        "${_FC_CACHE_DIR}"
        CACHE INTERNAL "")
    set(_FC_CACHE_PY
        "${CMAKE_CURRENT_SOURCE_DIR}/fetch_cache.py"
        CACHE INTERNAL "")
    set(_FC_PYTHON
        "${_FC_PYTHON}"
        CACHE INTERNAL "")
    message(STATUS "FetchContent cache: enabled (${_FC_CACHE_DIR})")
  endif()
endif()

# Override FetchContent_MakeAvailable; the original is auto-aliased to
# _FetchContent_MakeAvailable.  No-op when the cache is disabled.
macro(FetchContent_MakeAvailable)
  _fetchcontent_makeavailable(${ARGV})
  if(NOT "${_FC_CACHE_DIR}" STREQUAL "")
    foreach(_trtllm_fc_dep IN ITEMS ${ARGV})
      _trtllm_fc_update_one("${_trtllm_fc_dep}")
    endforeach()
  endif()
endmacro()

function(_trtllm_fc_update_one name)
  string(TOLOWER "${name}" _lc)
  if(NOT DEFINED ${_lc}_SOURCE_DIR)
    return()
  endif()
  execute_process(COMMAND "${_FC_PYTHON}" "${_FC_CACHE_PY}" update --cache-dir
                          "${_FC_CACHE_DIR}" --src "${${_lc}_SOURCE_DIR}")
endfunction()
# ── End FetchContent cache ───────────────────────────────────────────────

# Read dependencies from fetch_content.json
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/fetch_content.json" FETCH_CONTENT_JSON)
string(JSON DEP_COUNT LENGTH "${FETCH_CONTENT_JSON}" "dependencies")
math(EXPR DEP_COUNT_MINUS_ONE "${DEP_COUNT} - 1")

foreach(DEP_IDX RANGE ${DEP_COUNT_MINUS_ONE})
  # cmake-format: off
  # Extract dependency sub-object
  string(JSON DEP_OBJECT GET "${FETCH_CONTENT_JSON}" "dependencies" ${DEP_IDX})

  # Extract required fields
  string(JSON DEP_NAME     GET "${DEP_OBJECT}" "name")
  string(JSON DEP_GIT_REPO GET "${DEP_OBJECT}" "git_repository")
  string(JSON DEP_GIT_TAG  GET "${DEP_OBJECT}" "git_tag")

  # Expand github_base_url variable in git_repository if present
  string(REPLACE "\${github_base_url}" "${github_base_url}" DEP_GIT_REPO "${DEP_GIT_REPO}")

  # Extract optional fields with error handling
  string(JSON DEP_GIT_SHALLOW            ERROR_VARIABLE _err GET "${DEP_OBJECT}" "git_shallow")
  string(JSON DEP_SOURCE_SUBDIR          ERROR_VARIABLE _err GET "${DEP_OBJECT}" "source_subdir")
  string(JSON DEP_GIT_SUBMODULES_RECURSE ERROR_VARIABLE _err GET "${DEP_OBJECT}" "git_submodules_recurse")
  string(JSON DEP_USE_URL                ERROR_VARIABLE _err GET "${DEP_OBJECT}" "use_url")
  string(JSON DEP_PATCH_FILE            ERROR_VARIABLE _err GET "${DEP_OBJECT}" "patch_file")
  # cmake-format: on

  # Build FetchContent_Declare arguments
  set(FETCH_ARGS "${DEP_NAME}")

  # Handle URL vs GIT_REPOSITORY based on use_url flag
  if(DEP_USE_URL)
    list(APPEND FETCH_ARGS URL "${DEP_GIT_REPO}/archive/${DEP_GIT_TAG}.tar.gz")
  else()
    list(APPEND FETCH_ARGS GIT_REPOSITORY "${DEP_GIT_REPO}" GIT_TAG
         "${DEP_GIT_TAG}")
    if(DEP_GIT_SHALLOW)
      list(APPEND FETCH_ARGS GIT_SHALLOW TRUE)
    endif()
    if(DEP_GIT_SUBMODULES_RECURSE)
      list(APPEND FETCH_ARGS GIT_SUBMODULES_RECURSE ON)
    endif()
  endif()

  if(DEP_SOURCE_SUBDIR AND NOT DEP_SOURCE_SUBDIR STREQUAL "")
    list(APPEND FETCH_ARGS SOURCE_SUBDIR "${DEP_SOURCE_SUBDIR}")
  endif()

  if(DEP_PATCH_FILE AND NOT DEP_PATCH_FILE STREQUAL "")
    set(_patch_file "${CMAKE_CURRENT_SOURCE_DIR}/${DEP_PATCH_FILE}")
    list(
      APPEND
      FETCH_ARGS
      PATCH_COMMAND
      bash
      -c
      "patch -p1 --forward --batch --dry-run -i '${_patch_file}' && patch -p1 --forward --batch -i '${_patch_file}' || echo 'Patch already applied, skipping.'"
    )
  endif()

  FetchContent_Declare(${FETCH_ARGS})

  # Special handling: Export deep_ep commit to global property
  if(DEP_NAME STREQUAL "deep_ep_download")
    set_property(GLOBAL PROPERTY DEEP_EP_COMMIT "${DEP_GIT_TAG}")
  endif()
endforeach()
