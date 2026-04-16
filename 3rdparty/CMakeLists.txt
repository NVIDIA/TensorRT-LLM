include(ExternalProject)
include(FetchContent)

if(DEFINED ENV{GITHUB_MIRROR} AND NOT "$ENV{GITHUB_MIRROR}" STREQUAL "")
  set(github_base_url "$ENV{GITHUB_MIRROR}")
else()
  set(github_base_url "https://github.com")
endif()

# ── FetchContent cache acceleration ──────────────────────────────────────
# Maintains bare reference repos so that subsequent clean builds reuse
# already-downloaded git objects via git clone --reference.
#
# Cache directory:
#   $TRTLLM_FETCHCONTENT_CACHE  (env var, shared across repos)
#   or <3rdparty>/.cache_3rdparty  (default, per-repo)
#
# Set TRTLLM_FETCHCONTENT_CACHE="" to disable.
# ─────────────────────────────────────────────────────────────────────────
set(_FC_JSON "${CMAKE_CURRENT_SOURCE_DIR}/fetch_content.json")
set(_FC_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/fetch_cache.py")
set(_FC_WRAPPER "${CMAKE_CURRENT_SOURCE_DIR}/fetch_cache_wrapper.py")
set(_FC_ENABLED FALSE)

if(NOT WIN32
   AND EXISTS "${_FC_JSON}"
   AND EXISTS "${_FC_SCRIPT}"
   AND EXISTS "${_FC_WRAPPER}")

  # Check for explicit disable
  set(_FC_DISABLED FALSE)
  if(DEFINED ENV{TRTLLM_FETCHCONTENT_CACHE}
     AND "$ENV{TRTLLM_FETCHCONTENT_CACHE}" STREQUAL "")
    set(_FC_DISABLED TRUE)
  endif()

  if(NOT _FC_DISABLED)
    # Resolve cache directory
    if(DEFINED ENV{TRTLLM_FETCHCONTENT_CACHE}
       AND NOT "$ENV{TRTLLM_FETCHCONTENT_CACHE}" STREQUAL "")
      set(_FC_CACHE_DIR "$ENV{TRTLLM_FETCHCONTENT_CACHE}")
    else()
      set(_FC_CACHE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.cache_3rdparty")
    endif()

    # Save the real git before we override GIT_EXECUTABLE
    find_package(Git REQUIRED)
    set(_FC_REAL_GIT "${GIT_EXECUTABLE}")

    # Find Python for cache init script
    if(Python3_EXECUTABLE)
      set(_FC_PYTHON "${Python3_EXECUTABLE}")
    elseif(Python_EXECUTABLE)
      set(_FC_PYTHON "${Python_EXECUTABLE}")
    else()
      find_program(_FC_PYTHON NAMES python3 python)
    endif()

    if(_FC_PYTHON)
      # Init cache repos (parallel, with flock for concurrent safety)
      message(STATUS "FetchContent cache: initializing (${_FC_CACHE_DIR})")
      execute_process(
        COMMAND "${_FC_PYTHON}" "${_FC_SCRIPT}" init "${_FC_JSON}"
                --cache-dir "${_FC_CACHE_DIR}"
        RESULT_VARIABLE _fc_init_result)

      if(_fc_init_result EQUAL 0 AND IS_DIRECTORY "${_FC_CACHE_DIR}")
        # Inject env vars for the wrapper and override GIT_EXECUTABLE
        set(ENV{_TRTLLM_REAL_GIT} "${_FC_REAL_GIT}")
        set(ENV{TRTLLM_FETCHCONTENT_CACHE} "${_FC_CACHE_DIR}")
        set(GIT_EXECUTABLE "${_FC_WRAPPER}" CACHE FILEPATH
            "Git wrapper with reference cache acceleration" FORCE)
        set(_FC_ENABLED TRUE)
        message(STATUS "FetchContent cache: enabled")
      else()
        message(WARNING "FetchContent cache: init failed, proceeding without cache")
      endif()
    endif()
  endif()
endif()
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
