#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

struct EmitThreadIterationPass
    : public PassWrapper<EmitThreadIterationPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "emit-thread-iters"; }
  StringRef getDescription() const override {
    return "Emit a C program that counts per-thread iterations for each Triton kernel";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::error_code ec;
    llvm::raw_fd_ostream os("thread_iterations.c", ec);
    if (ec) {
      module.emitError("could not open thread_iterations.c: ") << ec.message();
      return;
    }
    os << "#include <stdint.h>\n";
    os << "#include <stdio.h>\n\n";
    // For all public function ops in the module (Triton kernel entry points)
    module.walk([&](func::FuncOp func) {
      if (!func.isPublic())
        return;
      auto funcType = func.getFunctionType();
      os << "void " << func.getName().str() << "_iter(\n";
      for (unsigned i = 0, n = funcType.getNumInputs(); i < n; ++i) {
        Type t = funcType.getInput(i);
        if (mlir::isa<mlir::MemRefType>(t)) {
          os << "    float *arg" << i; // Use actual element type if needed
        } else if (mlir::isa<mlir::IndexType>(t) || t.isIntOrIndex()) {
          os << "    int64_t arg" << i;
        } else {
          os << "    /* unsupported type */";
        }
        os << (i + 1 == n ? "\n" : ",\n");
      }
      os << ") {\n";
      os << "  // TODO: Insert thread/stride logic\n";
      os << "}\n\n";
    });
    os << "int main(int argc, char **argv) {\n";
    os << "  // TODO: Parse CLI args, call <kernel>_iter as above\n";
    os << "  return 0;\n";
    os << "}\n";
    os.close();
  }
};

static PassRegistration<EmitThreadIterationPass> pass;

std::unique_ptr<mlir::Pass> mlir::createEmitThreadIterationPass() {
    return std::make_unique<EmitThreadIterationPass>();
}

} // namespace mlir
