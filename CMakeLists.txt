cmake_minimum_required(VERSION 3.20)
project(toycalculator)

# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# Include MLIR CMake modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Set include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_DIR})

# Define TableGen targets for Toy_Dialect.td
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/src/Toy_Dialect.td)

# Generate operation declarations and definitions
mlir_tablegen(ToyDialect.h.inc -gen-op-decls -dialect toy)
mlir_tablegen(ToyDialect.cpp.inc -gen-op-defs -dialect toy)

# Generate type declarations and definitions
mlir_tablegen(ToyDialectTypes.h.inc -gen-typedef-decls -dialect toy)
mlir_tablegen(ToyDialectTypes.cpp.inc -gen-typedef-defs -dialect toy)

# Generate dialect declarations and definitions
mlir_tablegen(ToyDialectBase.h.inc -gen-dialect-decls -dialect toy)
mlir_tablegen(ToyDialectBase.cpp.inc -gen-dialect-defs -dialect toy)

# Create a custom target for TableGen outputs
add_public_tablegen_target(ToyDialectIncGen)

# Define the toycalculator executable
add_executable(toycalculator
  src/driver.cpp
  src/ToyCalculatorDialect.cpp
)

# Link against MLIR and LLVM libraries
target_link_directories(toycalculator PRIVATE /usr/local/llvm-20.1.2/lib64)
target_link_libraries(toycalculator PRIVATE
  LLVMSupport
  LLVMCore
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRParser
  MLIRTransforms
  MLIRBytecodeReader
  MLIRAsmParser
)

# Compile with -fno-rtti to match LLVM build
# c++20 for std::format()
target_compile_options(toycalculator PRIVATE
  -std=c++20
  -fno-rtti
  -Wall
  -Wextra
  -Wno-comment
  -Wno-overloaded-virtual
  -Wno-unused-parameter
  -g
)

# Ensure TableGen runs before building the executable
add_dependencies(toycalculator ToyDialectIncGen)
