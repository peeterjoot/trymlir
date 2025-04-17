#include "ToyDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"

// Define aliases for dependent dialects
using builtin = mlir::BuiltinDialect;
using arith = mlir::arith::ArithDialect;

// Include generated dialect definitions
#include "ToyDialectBase.cpp.inc"

MLIR_DEFINE_EXPLICIT_TYPE_ID( toy::AssignOp )
MLIR_DEFINE_EXPLICIT_TYPE_ID( toy::DeclOp )
MLIR_DEFINE_EXPLICIT_TYPE_ID( toy::PrintOp )
MLIR_DEFINE_EXPLICIT_TYPE_ID( toy::VarType )

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

    void ToyDialect::printType( mlir::Type type,
                                mlir::DialectAsmPrinter &printer ) const
    {
        if ( mlir::isa<VarType>( type ) )
        {
            printer << "var";
        }
    }

    llvm::LogicalResult AssignOp::readProperties(
        mlir::DialectBytecodeReader &reader, mlir::OperationState &state )
    {
        mlir::StringAttr nameAttr;
        if ( failed( reader.readAttribute( nameAttr ) ) )
            return llvm::failure();
        state.addAttribute( "name", nameAttr );
        return llvm::success();
    }

#if 0
    void DeclOp::build( mlir::OpBuilder &builder, mlir::OperationState &state,
                        llvm::StringRef name, mlir::Value value )
    {
        state.addAttribute( "name", builder.getStringAttr( name ) );
        state.addOperands( value );
    }

    void AssignOp::build( mlir::OpBuilder &builder, mlir::OperationState &state,
                          llvm::StringRef name, mlir::Value value )
    {
        state.addAttribute( "name", builder.getStringAttr( name ) );
        state.addOperands( value );
    }

    void PrintOp::build( mlir::OpBuilder &builder, mlir::OperationState &state,
                         llvm::StringRef name, mlir::Value value )
    {
        state.addAttribute( "name", builder.getStringAttr( name ) );
        state.addOperands( value );
    }

    void DeclOp::build( mlir::OpBuilder &builder, mlir::OperationState &state,
                        llvm::StringRef name )
    {
        state.addAttribute( "name", builder.getStringAttr( name ) );
    }

    void AssignOp::build( mlir::OpBuilder &builder, mlir::OperationState &state,
                          llvm::StringRef name )
    {
        state.addAttribute( "name", builder.getStringAttr( name ) );
    }

    void PrintOp::build( mlir::OpBuilder &builder, mlir::OperationState &state,
                         llvm::StringRef name )
    {
        state.addAttribute( "name", builder.getStringAttr( name ) );
    }
#endif

}    // namespace toy
