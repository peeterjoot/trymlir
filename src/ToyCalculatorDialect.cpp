#include "ToyDialect.h.inc"
#include "ToyDialectTypes.h.inc"
#include "mlir/IR/DialectRegistry.h"

namespace toy
{
    void ToyDialect::initialize()
    {
        addOperations<
#define GET_OP_LIST
#include "ToyDialect.cpp.inc"
            >();
        addTypes<
#define GET_TYPEDEF_LIST
#include "ToyDialectTypes.cpp.inc"
            >();
    }
}    // namespace toy

#include "ToyDialectBase.cpp.inc"

// vim: et ts=4 sw=4
