include(ExternalProject)
include(FetchContent)

if(DEFINED ENV{GITHUB_MIRROR} AND NOT "$ENV{GITHUB_MIRROR}" STREQUAL "")
  set(github_base_url "$ENV{GITHUB_MIRROR}")
else()
  set(github_base_url "https://github.com")
endif()

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

  FetchContent_Declare(${FETCH_ARGS})

  # Special handling: Export deep_ep commit to global property
  if(DEP_NAME STREQUAL "deep_ep_download")
    set_property(GLOBAL PROPERTY DEEP_EP_COMMIT "${DEP_GIT_TAG}")
  endif()
endforeach()
