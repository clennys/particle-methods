cmake_minimum_required(VERSION 3.16)
project(Ising CXX)
include(FetchContent)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FetchContent_Declare(matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
        GIT_TAG origin/master)

FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
    FetchContent_Populate(matplotplusplus)
    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Include directory
include_directories(${CMAKE_SOURCE_DIR}/include)

# Collect all source and header files
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE HEADERS ${CMAKE_SOURCE_DIR}/include/*.hpp)

# Create executable or library
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link Matplot++ (Even though it's header-only, this ensures compatibility)
target_link_libraries(${PROJECT_NAME} PUBLIC matplot)

