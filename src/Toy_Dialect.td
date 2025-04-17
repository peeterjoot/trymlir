include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"

// Define the Toy dialect
def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "Toy dialect for a simple calculator";
  let description = [{
    The Toy dialect models a simple calculator language with variable declarations,
    assignments, and print operations.
  }];
  let cppNamespace = "::toy";
  let dependentDialects = ["builtin", "arith"];
  let extraClassDeclaration = [{
    void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const;
  }];
}

// Define a custom type for Toy variables
def ToyVarType : TypeDef<Toy_Dialect, "Var"> {
  let summary = "Type for named variables in the Toy dialect";
  let mnemonic = "var";
  let cppClassName = "VarType";
}

// Define operations

// Operation: toy.decl (DCL x;)
def Toy_DeclOp : Op<Toy_Dialect, "decl"> {
  let summary = "Declares a variable with a given name";
  let description = [{
    Declares a variable with a given name, producing a value of ToyVarType.
    Example: `toy.decl @x : !toy.var`
  }];

  let arguments = (ins SymbolNameAttr:$name);

  let results = (outs ToyVarType:$result);

  let assemblyFormat = [{
    $name `:` type($result) attr-dict
  }];

  let traits = [Pure];

  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name, "::mlir::Value":$value)>
  ];
}

// Operation: toy.assign (x = 5;)
def Toy_AssignOp : Op<Toy_Dialect, "assign"> {
  let summary = "Assigns a value to a variable";
  let description = [{
    Assigns a value (e.g., an integer) to a variable.
    Example: `toy.assign @x, %value : i32`
  }];

  let arguments = (ins
    SymbolNameAttr:$varName,
    AnyTypeOf<[I32, ToyVarType]>:$value
  );

  let results = (outs);

  let assemblyFormat = [{
    $varName `,` $value `:` type($value) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name, "::mlir::Value":$value)>
  ];
}

// Operation: toy.print (PRINT x;)
def Toy_PrintOp : Op<Toy_Dialect, "print"> {
  let summary = "Prints a variable's value";
  let description = [{
    Prints the value of a variable.
    Example: `toy.print @x, %0 : !toy.var`
  }];

  let arguments = (ins
    SymbolNameAttr:$varName,
    ToyVarType:$var
  );

  let results = (outs);

  let assemblyFormat = [{
    $varName `,` $var `:` type($var) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name, "::mlir::Value":$value)>
  ];
}
