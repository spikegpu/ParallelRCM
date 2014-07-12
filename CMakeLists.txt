CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(rcm)
INCLUDE(cmake/SBELUtils.cmake)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

enable_cuda_support()
MESSAGE(STATUS "Cuda version: ${CUDA_VERSION}")

# CUSP dependency. Note that we always include CUSP headers using 
# include <cusp/header.h>.
find_path(CUSP_INCLUDE_DIR 
          NAMES cusp/array1d.h 
          HINTS ${CUDA_TOOLKIT_INCLUDE} 
          DOC "Top-level CUSP directory")
if(CUSP_INCLUDE_DIR)
   message(STATUS "CUSP headers found in ${CUSP_INCLUDE_DIR}/cusp")
else()
   message(FATAL_ERROR "CUSP headers not found. Please set CUSP_INCLUDE_DIR.")
endif()
cuda_include_directories(${CUSP_INCLUDE_DIR})

SET(RCM_HEADERS
	rcm/common.h
	rcm/rcm.h
	rcm/rcm_um.h
	rcm/exception.h
	rcm/timer.h
  )

SET(RCM_CUHEADERS
  rcm/device/kernels.cuh
  )

SET(MMIO_FILES
  mm_io/mm_io.h
  mm_io/mm_io.c
  )

SOURCE_GROUP("Headers" FILES ${RCM_HEADERS})
SOURCE_GROUP("CUDA Headers" FILES ${RCM_CUHEADERS})
SOURCE_GROUP("MM_IO" FILES ${MMIO_FILES})

INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}
    )

IF(NOT (${CUDA_VERSION} VERSION_LESS "6.0"))
  cuda_add_executable(driver_um driver_um.cu ${RCM_HEADERS} ${RCM_CUHEADERS} ${MMIO_FILES})
  cuda_add_executable(testing testing.cu ${RCM_HEADERS} ${RCM_CUHEADERS} ${MMIO_FILES})
  cuda_add_executable(testing_um testing_um.cu ${RCM_HEADERS} ${RCM_CUHEADERS} ${MMIO_FILES})
ENDIF()
