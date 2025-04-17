#include "ToyDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/TypeID.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

// Define aliases for dependent dialects
using builtin = mlir::BuiltinDialect;
using arith = mlir::arith::ArithDialect;

// Include generated dialect definitions
#include "ToyDialectBase.cpp.inc"

MLIR_DEFINE_EXPLICIT_TYPE_ID(toy::AssignOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(toy::DeclOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(toy::PrintOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(toy::VarType)

namespace toy {

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ToyDialect.cpp.inc"
  >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "ToyDialectTypes.cpp.inc"
  >();
}

void ToyDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  if (type.isa<VarType>()) {
    printer << "var";
  }
}

} // namespace toy
