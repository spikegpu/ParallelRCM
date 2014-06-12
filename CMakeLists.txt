CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(rcm)
INCLUDE(cmake/SBELUtils.cmake)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

enable_cuda_support()
MESSAGE(STATUS "Cuda version: ${CUDA_VERSION}")

SET(RCM_HEADERS
	rcm/common.h
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

SOURCE_GROUP("Headers" FILES ${MC64_HEADERS})
SOURCE_GROUP("CUDA Headers" FILES ${MC64_CUHEADERS})
SOURCE_GROUP("MM_IO" FILES ${MMIO_FILES})

INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}
    )

IF(NOT (${CUDA_VERSION} VERSION_LESS "6.0"))
  cuda_add_executable(driver_um driver_um.cu ${MC64_HEADERS} ${MC64_CUHEADERS} ${MMIO_FILES})
  cuda_add_executable(testing testing.cu ${MC64_HEADERS} ${MC64_CUHEADERS} ${MMIO_FILES})
ENDIF()
