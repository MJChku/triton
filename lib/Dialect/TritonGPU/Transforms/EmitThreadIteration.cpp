//===----------------------------------------------------------------------===//
//  TritonGPUEmitThreadIteration
//  Prune IR to only ops feeding scf.for loops, then emit a C stub with bounds.
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUEMITTHREADITERATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct TritonGPUEmitThreadIterationPass
    : public impl::TritonGPUEmitThreadIterationBase<TritonGPUEmitThreadIterationPass> {
  void runOnOperation() override {

    return;
    
    ModuleOp module = getOperation();

    // 1) Clone and prune IR to only ops influencing scf.for loops
    ModuleOp pruned = getOperation(); //module.clone();
    pruned.walk([&](FuncOp fn) {
      // gather loops
      // SmallVector<Operation *> loops;
      // fn.walk([&](scf::ForOp op) { loops.push_back(op); });
      SmallVector<scf::ForOp> loops;
      fn.walk([&](scf::ForOp op) { loops.push_back(op); });

    // Compute the keep set: just the for op and all ops needed for its bounds
      llvm::DenseSet<Operation *> keep;

      fn.walk([&](Operation *op) {
        if (op->hasTrait<mlir::OpTrait::IsTerminator>()){
          keep.insert(op);
          llvm::errs() << "Keeping terminator: " << *op << "\n";
        }
      });

      for (scf::ForOp loopOp : loops) {
        // For each bound (lb, ub, step)
        for (Value v : {loopOp.getLowerBound(), loopOp.getUpperBound(), loopOp.getStep()}) {
          llvm::SetVector<Operation *> slice;
          getBackwardSlice(v, &slice);
          keep.insert(slice.begin(), slice.end());
        }

        keep.insert(loopOp.getOperation()); // Always keep the for op itself
      }

      // fn.walk([&](mlir::func::ReturnOp ret) {
      //   keep.insert(ret.getOperation());
      //   for (Value v : ret.getOperands()) {
      //     llvm::SetVector<Operation *> slice;
      //     getBackwardSlice(v, &slice);
      //     keep.insert(slice.begin(), slice.end());
      //   }
      // });

      if (!keep.empty()) {
        keep.insert(fn); 
      }

    
      SmallVector<Operation *> eraseList;
      fn.walk<WalkOrder::PostOrder>([&](Operation *op) {
        if (!keep.contains(op)){
          eraseList.push_back(op);
          llvm::errs() << "Marking for erase: " << *op << "\n";
        }
      });

      llvm::DenseSet<Operation *> toErase(eraseList.begin(), eraseList.end());

      bool removedSomething = true;
      while (removedSomething) {
        removedSomething = false;
        for (size_t i = 0; i < eraseList.size();) {
          Operation *op = eraseList[i];
          if (op->use_empty()) {
            llvm::errs() << "Erasing: " << *op << "\n";
            op->erase();
            eraseList.erase(eraseList.begin() + i);
            removedSomething = true;
          } else {
            ++i;
          }
        }
      }


      SmallVector<Block*> allBlocks;
      std::function<void(Region&)> collectBlocks = [&](Region &r) {
        for (Block &b : r) {
          allBlocks.push_back(&b);
          for (Operation &op : b) {
            for (Region &nestedRegion : op.getRegions()) {
              collectBlocks(nestedRegion); // recursive
            }
          }
        };
      };

      collectBlocks(fn.getBody());

      for (Block *b : allBlocks) {
        if (b->getOperations().size() == 1) {
          Operation &onlyOp = b->front();
          if (onlyOp.hasTrait<OpTrait::IsTerminator>()) {
            onlyOp.erase();
          }
        }
      }


    });
  }
};

std::unique_ptr<Pass> createTritonGPUEmitThreadIterationPass() {
  return std::make_unique<TritonGPUEmitThreadIterationPass>();
}

} // namespace gpu
} // namespace triton
} // namespace mlir
