CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project(elephant LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

find_package(ArrayFire REQUIRED)
find_package(OpenCL REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories(${ArrayFire_INCLUDE_DIRS})

file(GLOB_RECURSE SOURCES src/*.cpp)
add_library(elephant SHARED ${SOURCES})
target_link_libraries(elephant ${ArrayFire_OpenCL_LIBRARIES} ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

file(GLOB EXECS tools/*)
foreach(exe_target IN ITEMS ${EXECS})
        file(GLOB_RECURSE SOURCES ${exe_target}/*.cpp)
        get_filename_component(exe_name ${exe_target} NAME)
        add_executable(${exe_name} ${SOURCES})
	target_link_libraries(${exe_name} elephant)
endforeach(exe_target)
