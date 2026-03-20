if (ENABLE_CUDA)
    file(GLOB CUDA_SOURCE_FILES "cuda/*.cu")
endif()

file(GLOB SOURCE_FILES "*.cpp")

# SIMD-optimized source files
file(GLOB SIMD_SOURCE_FILES "simd/*.cpp")

# Create the library
add_library(${PROJECT_NAME} ${SOURCE_FILES} ${SIMD_SOURCE_FILES} ${CUDA_SOURCE_FILES})
