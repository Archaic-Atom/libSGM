cmake_minimum_required(VERSION 3.1)

find_package(CUDA REQUIRED)

if(DEFINED AUTO_DETECT_ARCH AND "${AUTO_DETECT_ARCH}")
	CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
	set(CUDA_ARCH "${ARCH_FLAGS}" CACHE STRING "Value of the NVCC -arch option." FORCE)
endif()
message(STATUS "CUDA_ARCH: \"${CUDA_ARCH}\"")

include_directories(../include)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")

find_package(OpenCV REQUIRED)
if (OpenCV_VERSION VERSION_LESS 3.0)
	message(FATAL_ERROR "Error: OpenCV version requires at least 3.0")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(../../include)
include_directories(./)
include_directories(/usr/include/python3.6)

file(GLOB SGM_SOURCE "../../src/*.cu" "../../src/*.cpp")


cuda_add_library(sgm_interface_fast_system SHARED sgm_interface_fast_system.cpp ${SGM_SOURCE} ${CUDA_SRC})
target_link_libraries(sgm_interface_fast_system ${CUDA_LIBRARIES} ${OpenCV_LIBS})