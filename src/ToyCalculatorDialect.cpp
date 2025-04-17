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

#if 0
void ToyDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  if (type.isa<VarType>()) {
    printer << "var";
  }
}
#endif

} // namespace toy
