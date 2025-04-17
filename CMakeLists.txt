cmake_minimum_required(VERSION 3.20)
project(toycalculator)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/src/Toy_Dialect.td)

# Generate operation declarations and definitions (intermediate file)
mlir_tablegen(ToyDialect.h.inc -gen-op-decls -dialect toy --op-exclude-regex "arith.*")
mlir_tablegen(ToyDialect.cpp.inc -gen-op-defs -dialect toy --op-exclude-regex "arith.*")

# Patch ToyDialect.h.inc.gen to remove TypeID macros and produce ToyDialect.h.inc
#add_custom_command(
#  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ToyDialect.h.inc
#  COMMAND grep -v ^MLIR_DECLARE_EXPLICIT_TYPE_ID ${CMAKE_CURRENT_BINARY_DIR}/ToyDialect.h.inc.gen > ${CMAKE_CURRENT_BINARY_DIR}/ToyDialect.h.inc
#  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ToyDialect.h.inc.gen
#  COMMENT "Patching ToyDialect.h.inc to remove TypeID macros"
#  VERBATIM
#)

# Custom target for the patching step
#add_custom_target(ToyDialectPatch DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ToyDialect.h.inc)

# Generate type declarations and definitions
mlir_tablegen(ToyDialectTypes.h.inc -gen-typedef-decls -dialect toy)
mlir_tablegen(ToyDialectTypes.cpp.inc -gen-typedef-defs -dialect toy)

# Generate dialect declarations and definitions
mlir_tablegen(ToyDialectBase.h.inc -gen-dialect-decls -dialect toy)
mlir_tablegen(ToyDialectBase.cpp.inc -gen-dialect-defs -dialect toy)

# Create a custom target for TableGen outputs
add_public_tablegen_target(ToyDialectIncGen)

# Ensure ToyDialectIncGen depends on the patching step
#add_dependencies(ToyDialectIncGen ToyDialectPatch)

# Mark ToyDialect.h.inc as generated
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/ToyDialect.h.inc PROPERTIES GENERATED TRUE)

add_executable(toycalculator
  src/driver.cpp
  src/ToyCalculatorDialect.cpp
)

target_link_directories(toycalculator PRIVATE /usr/local/llvm-20.1.3/lib64)
target_link_libraries(toycalculator PRIVATE
  LLVMSupport
  LLVMCore
  MLIRIR
  MLIRSupport
  MLIRParser
  MLIRTransforms
  MLIRBytecodeReader
  MLIRAsmParser
  MLIRArith
)

target_compile_options(toycalculator PRIVATE
  -Wall
  -Wextra
  -Wno-comment
  -Wno-overloaded-virtual
  -Wno-unused-parameter
  -g
  -fno-rtti
)

add_dependencies(toycalculator ToyDialectIncGen)
