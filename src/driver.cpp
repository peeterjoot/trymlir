#include <format>
#include <iostream>
#include <string>

#include "ToyDialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc( "<input file>" ),
    llvm::cl::init( "-" ), llvm::cl::value_desc( "filename" ) );

// Fake location info function (replace with your implementation if needed)
// mlir::Location getLocation(int line) {
//  return mlir::UnknownLoc::get(&context);
//}
//#define getLocation( line ) \
//    mlir::FileLineColLoc::get( builder.getStringAttr( inputFilename ), line, 1 )
#define getLocation( line ) \
    builder.getUnknownLoc()

int main( int argc, char **argv )
{
    llvm::InitLLVM init( argc, argv );
    llvm::cl::ParseCommandLineOptions( argc, argv, "Calculator compiler\n" );

    std::cout << std::format( "Processing fake file named '{}'\n", inputFilename.c_str() );

    mlir::MLIRContext context;
    context.getOrLoadDialect<toy::ToyDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::ModuleOp::create( mlir::UnknownLoc::get( &context ) );
    mlir::OpBuilder builder( &context );
    builder.setInsertionPointToStart( module->getBody() );

    // Create toy.decl @x
    auto declOp = builder.create<toy::DeclOp>( getLocation( 1 ), "x" );
    auto var = declOp.getResult();

    // Create toy.assign @x, 5
    auto constOp = builder.create<mlir::arith::ConstantOp>(
        getLocation( 2 ), builder.getI32IntegerAttr( 5 ) );
    builder.create<toy::AssignOp>( getLocation( 2 ), "x", constOp.getResult() );

    // Create toy.print @x
    builder.create<toy::PrintOp>( getLocation( 3 ), "x", var );

    builder.setInsertionPointToEnd( module->getBody() );

    // Print the module using std::format
    std::string moduleStr;
    llvm::raw_string_ostream os( moduleStr );
    module->print( os );
    std::cout << std::format( "Generated MLIR module:\n{}\n", moduleStr );

    return 0;
}

// vim: et ts=4 sw=4
