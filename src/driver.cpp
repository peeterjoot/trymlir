#include "ToyDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include <format>
#include <iostream>
#include <string>

// Fake location info function (replace with your implementation if needed)
mlir::Location getLocation(int line) {
  return mlir::UnknownLoc::get(&context);
}
//#define getLocation(line) \
//    mlir::FileLineColLoc::get(builder.getStringAttr(inputFilename), line, 1)

int main(int argc, char **argv) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<toy::ToyDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToStart(module->getBody());

  // Create toy.decl @x
  auto declOp = builder.create<toy::DeclOp>(getLocation(1), "x", toy::VarType::get(&context));
  auto var = declOp.getResult();

  // Create toy.assign @x, 5
  auto constOp = builder.create<mlir::arith::ConstantOp>(
      getLocation(2), builder.getI32IntegerAttr(5));
  builder.create<toy::AssignOp>(getLocation(2), "x", constOp.getResult());

  // Create toy.print @x
  builder.create<toy::PrintOp>(getLocation(2), "x", var);

  // Print the module using std::format
  std::string moduleStr;
  llvm::raw_string_ostream os(moduleStr);
  module->print(os);
  std::cout << std::format("Generated MLIR module:\n{}\n", moduleStr);

  return 0;
}

// vim: et ts=4 sw=4
