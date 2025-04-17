// Toy_Dialect.td
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

  // Arguments: variable name as a symbol
  let arguments = (ins SymbolNameAttr:$name);

  // Results: produces a ToyVarType
  let results = (outs ToyVarType:$result);

  // Assembly format
  let assemblyFormat = [{
    $name `:` type($result) attr-dict
  }];

  // No side effects (pure operation)
  let traits = [Pure];

  // Extra C++ declaration
  let extraClassDeclaration = [{
    auto getOperationName() const { return "toy.decl"; }
  }];
}

// Operation: toy.assign (x = 5;)
def Toy_AssignOp : Op<Toy_Dialect, "assign"> {
  let summary = "Assigns a value to a variable";
  let description = [{
    Assigns a value (e.g., an integer) to a variable.
    Example: `toy.assign @x, %value : i32`
  }];

  // Arguments: variable name (SymbolNameAttr), value (e.g., i32 or ToyVarType)
  let arguments = (ins
    SymbolNameAttr:$varName,
    AnyTypeOf<[I32, ToyVarType]>:$value
  );

  // Results: none (void operation)
  let results = (outs);

  // Assembly format
  let assemblyFormat = [{
    $varName `,` $value `:` type($value) attr-dict
  }];
}

// Operation: toy.print (PRINT x;)
def Toy_PrintOp : Op<Toy_Dialect, "print"> {
  let summary = "Prints a variable's value";
  let description = [{
    Prints the value of a variable.
    Example: `toy.print @x : !toy.var`
  }];

  // Arguments: variable name (SymbolNameAttr), variable (ToyVarType)
  let arguments = (ins
    SymbolNameAttr:$varName,
    ToyVarType:$var
  );

  // Results: none (void operation)
  let results = (outs);

  // Assembly format
  let assemblyFormat = [{
    $varName `,` $var `:` type($var) attr-dict
  }];
}
