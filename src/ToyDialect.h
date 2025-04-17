#ifndef TOY_DIALECT_H
#define TOY_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/TypeID.h"

// Include generated dialect declarations
#include "ToyDialectBase.h.inc"
// Include generated type declarations
#include "ToyDialectTypes.h.inc"
// Include generated operation declarations
#include "ToyDialect.h.inc"

namespace toy {

// Forward declaration of ToyDialect
class ToyDialect;

// Define types and operations using TableGen includes
#define GET_OP_CLASSES
#include "ToyDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ToyDialectTypes.h.inc"

} // namespace toy

#endif // TOY_DIALECT_H
