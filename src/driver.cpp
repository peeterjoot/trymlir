#include <format>
#include <fstream>
#include <iostream>

#include "ToyDialect.h.inc"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc( "<input file>" ),
    llvm::cl::init( "-" ), llvm::cl::value_desc( "filename" ) );

#if 0
void registerToyDialect( mlir::DialectRegistry &registry )
{
    registry.insert<toy::ToyDialect>();
}
#endif

// FIXME: later make this a member function of the Parse Listener class, with private builder and filename member variables, and get the
// location info from the parser context instead of made up:
#define getLocation(line) \
    mlir::FileLineColLoc::get(builder.getStringAttr(inputFilename), line, 1)

int main( int argc, char **argv )
{
    llvm::InitLLVM init( argc, argv );
    llvm::cl::ParseCommandLineOptions( argc, argv, "Calculator compiler\n" );
    mlir::MLIRContext context;
    context.getOrLoadDialect<toy::ToyDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::ModuleOp::create( mlir::UnknownLoc::get( &context ) );
    mlir::OpBuilder builder( &context );
    builder.setInsertionPointToStart( module->getBody() );

    // Create toy.decl @x
    auto declOp = builder.create<toy::DeclOp>( getLocation(1), "x",
                                               toy::VarType::get( &context ) );
    auto var = declOp.getResult();

    // Create toy.assign @x, 5
    auto constOp = builder.create<mlir::arith::ConstantIntOp>(
        getLocation(2), 5, 32 );
    builder.create<toy::AssignOp>( getLocation(2), "x",
                                   constOp.getResult() );

    // Create toy.print @x
    builder.create<toy::PrintOp>( getLocation(2), "x", var );

    builder.setInsertionPointToEnd(module.getBody());

    // Print the module
    module->print( llvm::outs() );
    return 0;
}

// vim: et ts=4 sw=4
