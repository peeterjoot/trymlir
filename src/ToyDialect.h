#ifndef TOY_DIALECT_H
#define TOY_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"

// Include the generated type declarations
#define GET_TYPEDEF_CLASSES
#include "ToyDialectTypes.h.inc"

// Include the generated dialect declarations
#include "ToyDialectBase.h.inc"

// Include the generated operation declarations
#define GET_OP_CLASSES
#include "ToyDialect.h.inc"

#endif // TOY_DIALECT_H
