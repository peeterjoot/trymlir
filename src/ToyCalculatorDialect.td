include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "Toy dialect for a simple calculator";
  let cppNamespace = "::toy";
}
