#include "ToyDialect.h"

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

ToyDialect::ToyDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<ToyDialect>()) {
  initialize();
}

} // namespace toy

#include "ToyDialectBase.cpp.inc"
