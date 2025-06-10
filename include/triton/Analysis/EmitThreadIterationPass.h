#pragma once

#include <memory>
#include "mlir/Pass/Pass.h"

namespace mlir {

/// Creates a pass that emits a C program to count per-thread iterations
/// for each Triton kernel function in the module.
std::unique_ptr<mlir::Pass> createEmitThreadIterationPass();

} // namespace mlir
