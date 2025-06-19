#include "Dialect/NVGPU/IR/Dialect.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TAINTANALYSISONLLVM
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::NVIDIA;

namespace {

struct TaintAnalysisOnLLVM
    : public triton::impl::TaintAnalysisOnLLVMBase<TaintAnalysisOnLLVM> {
  using TaintAnalysisOnLLVMBase::TaintAnalysisOnLLVMBase;

private:
  // Track values that are "tainted" (come from dummy GPU operations or global loads)
  llvm::DenseSet<Value> taintedValues;

   // Track which memory regions are tainted
  llvm::DenseSet<Value> taintedMemoryRegions;  // base pointers
  llvm::DenseSet<Value> taintedAllocations;    // allocations containing tainted data
  
  // Map block arguments to their source values
  llvm::DenseMap<Value, llvm::SmallVector<Value>> blockArgSources;
  
  Value getBasePointer(Value ptr) {
    // Trace back through GEP chains to find the base allocation
    Value current = ptr;
    while (auto gep = current.getDefiningOp<LLVM::GEPOp>()) {
      current = gep.getBase();
    }
    return current;
  }

  void markMemoryRegionAsTainted(Value basePtr) {
    taintedMemoryRegions.insert(basePtr);
    llvm::errs() << "Marking memory region as tainted: " << basePtr << "\n";
  }
  
  bool isMemoryRegionTainted(Value ptr) {
    // Check if this pointer derives from a tainted memory region
    if (auto gep = ptr.getDefiningOp<LLVM::GEPOp>()) {
      Value base = gep.getBase();
      return taintedMemoryRegions.contains(base) || isMemoryRegionTainted(base);
    }
    return taintedMemoryRegions.contains(ptr);
  }

  void markValueAsTainted(Value val) {
    taintedValues.insert(val);
    llvm::errs() << "Marking value as tainted: " << val << "\n";
  }
  
  bool isValueTainted(Value val) {
    return taintedValues.contains(val);
  }
  
  void mapBlockArguments() {
    ModuleOp module = getOperation();
    
    module.walk([&](Operation *op) {
      // Track branch operations that pass values to blocks
      if (auto brOp = dyn_cast<LLVM::BrOp>(op)) {
        Block *destBlock = brOp.getDest();
        auto args = brOp.getDestOperands();
        
        for (auto [blockArg, sourceValue] : llvm::zip(destBlock->getArguments(), args)) {
          blockArgSources[blockArg].push_back(sourceValue);
        }
      }
      
      if (auto condBrOp = dyn_cast<LLVM::CondBrOp>(op)) {
        // True branch
        Block *trueBlock = condBrOp.getTrueDest();
        auto trueArgs = condBrOp.getTrueDestOperands();
        for (auto [blockArg, sourceValue] : llvm::zip(trueBlock->getArguments(), trueArgs)) {
          blockArgSources[blockArg].push_back(sourceValue);
        }
        
        // False branch  
        Block *falseBlock = condBrOp.getFalseDest();
        auto falseArgs = condBrOp.getFalseDestOperands();
        for (auto [blockArg, sourceValue] : llvm::zip(falseBlock->getArguments(), falseArgs)) {
          blockArgSources[blockArg].push_back(sourceValue);
        }
      }
    });
  }
  
  // Check if a value transitively depends on tainted values
  bool dependsOnTaintedData(Value val) {
    llvm::SmallVector<Value> worklist{val};
    llvm::DenseSet<Value> visited;
    
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      
      if (visited.contains(current))
        continue;
      visited.insert(current);
      
      if (isValueTainted(current)) {
        markValueAsTainted(val);
        return true;
      }
      
      // Check if this is a block argument
      if (mlir::isa<BlockArgument>(current)) {
        if (blockArgSources.contains(current)) {
          // Add all source values to worklist
          for (Value source : blockArgSources[current]) {
            worklist.push_back(source);
          }
        }
        continue; // Don't call getDefiningOp() on block arguments
      }
      
      if (Operation *defOp = current.getDefiningOp()) {
        // Add all operands to worklist for transitive checking
        for (Value operand : defOp->getOperands()) {
          worklist.push_back(operand);
        }
      }
    }
    
    return false;
  }

  void identifyTaintedValues() {
    ModuleOp module = getOperation();

     module.walk([&](LLVM::StoreOp storeOp) {
      Value storedValue = storeOp.getValue();
      Value ptr = storeOp.getAddr();
      
      // If storing a tainted value, mark the memory region as tainted
      if (isValueTainted(storedValue) || dependsOnTaintedData(storedValue)) {
        // Find the base allocation/pointer
        Value basePtr = getBasePointer(ptr);
        markMemoryRegionAsTainted(basePtr);
        
        // Also mark the specific allocation
        if (auto allocaOp = basePtr.getDefiningOp<LLVM::AllocaOp>()) {
          taintedAllocations.insert(basePtr);
        }
      }
    });
    
    module.walk([&](LLVM::CallOp callOp) {
      if (callOp.getCallee()) {
        StringRef calleeName = *callOp.getCallee();
        if (calleeName.contains("metrics_dummy")
          || calleeName.contains("undef")
          || calleeName.contains("unrealized_conversion_cast")
          ) {
          
          for (Value result : callOp.getResults()) {
            markValueAsTainted(result);
          }
        }
      }
    });
    
    // Mark loads from global memory as tainted
    module.walk([&](LLVM::LoadOp loadOp) {
      Value addr = loadOp.getAddr();

      Value basePtr = getBasePointer(addr);
      
      // If loading from a tainted memory region, mark the result as tainted
      if (isMemoryRegionTainted(basePtr)) {
        markValueAsTainted(loadOp.getResult());
        llvm::errs() << "Load from tainted memory: " << loadOp.getResult() << "\n";
      }
      
      // Check if loading from a global that might contain dummy data
      if (auto gep = basePtr.getDefiningOp<LLVM::GEPOp>()) {
        if (auto addrOf = gep.getBase().getDefiningOp<LLVM::AddressOfOp>()) {
          StringRef globalName = addrOf.getGlobalName();
          if (globalName.contains("global_smem")) {
            markValueAsTainted(loadOp.getResult());
          }
        }
      }
    });

    // Propagate taint through operations
    bool changed = true;
    while (changed) {
      changed = false;
      
      module.walk([&](Operation *op) {
        // Skip already processed operations
        if (isa<LLVM::CallOp>(op) || isa<LLVM::LoadOp>(op))
          return;
        
        // Check if any operand is tainted
        bool hasaTaintedOperand = false;
        for (Value operand : op->getOperands()) {
          if (isValueTainted(operand) || dependsOnTaintedData(operand)) {
            hasaTaintedOperand = true;
            break;
          }
        }
        
        // If so, mark all results as tainted
        if (hasaTaintedOperand) {
          for (Value result : op->getResults()) {
            if (!isValueTainted(result)) {
              markValueAsTainted(result);
              changed = true;
            }
          }
        }
      });
    }
  }

  void insertBranchAssertions() {
    ModuleOp module = getOperation();
    auto *ctx = &getContext();
    
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(module.getBody());
    
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, /*addrSpace=*/0);
    auto assertFnTy = LLVM::LLVMFunctionType::get(voidTy, /*params=*/{ptrTy}, /*isVarArg=*/false);
    
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("branch_assertion_failure")) {
      auto assertStub = builder.create<LLVM::LLVMFuncOp>(
          module.getLoc(), "branch_assertion_failure", assertFnTy);
      assertStub.setLinkage(LLVM::Linkage::External);
    }
    
    int unsafeBranches = 0;
    
    // Walk all conditional branches and check their conditions
    module.walk([&](LLVM::CondBrOp condBr) {
      Value condition = condBr.getCondition();
      
      // Check if the branch condition depends on tainted data
      if (dependsOnTaintedData(condition)) {
        unsafeBranches++;
        Location loc = condBr.getLoc();
        
        // Create error message string
        std::string errorMsg = "Branch " + std::to_string(unsafeBranches) + 
                              " depends on dummy GPU data or potentially wrong global memory";
        
        // Create global string for error message
        auto savedInsertionPoint = builder.saveInsertionPoint();
        builder.setInsertionPointToStart(module.getBody());
        
        auto stringTy = LLVM::LLVMArrayType::get(builder.getI8Type(), errorMsg.length() + 1);
        auto errorGlobal = builder.create<LLVM::GlobalOp>(
            loc, stringTy, /*isConstant=*/true, LLVM::Linkage::Private,
            "branch_error_" + std::to_string(unsafeBranches),
            builder.getStringAttr(errorMsg + '\0'));
        
        builder.restoreInsertionPoint(savedInsertionPoint);
        
        // Insert assertion check before the conditional branch
        builder.setInsertionPoint(condBr);
        
        auto errorPtr = builder.create<LLVM::AddressOfOp>(loc, ptrTy, errorGlobal.getSymName());
        
        // Create assertion call
        builder.create<LLVM::CallOp>(
            loc, /*resultTypes=*/TypeRange{},
            /*callee=*/builder.getStringAttr("branch_assertion_failure"),
            /*args=*/ValueRange{errorPtr});
        
        llvm::errs() << "WARNING: Inserted assertion for potentially unsafe branch at " 
                     << loc << "\n";
      }
    });
    
    // Also check switch operations
    module.walk([&](LLVM::SwitchOp switchOp) {
      Value condition = switchOp.getValue();
      
      if (dependsOnTaintedData(condition)) {
        unsafeBranches++;
        Location loc = switchOp.getLoc();
        
        std::string errorMsg = "Switch " + std::to_string(unsafeBranches) + 
                              " depends on dummy GPU data or potentially wrong global memory";
        
        // Create global string for error message
        auto savedInsertionPoint = builder.saveInsertionPoint();
        builder.setInsertionPointToStart(module.getBody());
        
        auto stringTy = LLVM::LLVMArrayType::get(builder.getI8Type(), errorMsg.length() + 1);
        auto errorGlobal = builder.create<LLVM::GlobalOp>(
            loc, stringTy, /*isConstant=*/true, LLVM::Linkage::Private,
            "switch_error_" + std::to_string(unsafeBranches),
            builder.getStringAttr(errorMsg + '\0'));
        
        builder.restoreInsertionPoint(savedInsertionPoint);
        
        builder.setInsertionPoint(switchOp);
        auto errorPtr = builder.create<LLVM::AddressOfOp>(loc, ptrTy, errorGlobal.getSymName());
        
        builder.create<LLVM::CallOp>(
            loc, /*resultTypes=*/TypeRange{},
            /*callee=*/builder.getStringAttr("branch_assertion_failure"),
            /*args=*/ValueRange{errorPtr});
        
        llvm::errs() << "WARNING: Inserted assertion for potentially unsafe switch at " 
                     << loc << "\n";
      }
    });
    
    if (unsafeBranches > 0) {
      llvm::errs() << "TAINT ANALYSIS: Found " << unsafeBranches 
                   << " potentially unsafe conditional operations\n";
    } else {
      llvm::errs() << "TAINT ANALYSIS: No unsafe conditional operations found\n";
    }
  }

  void insertMemoryAccessAssertions() {
    ModuleOp module = getOperation();
    auto *ctx = &getContext();
    
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(module.getBody());
    
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, /*addrSpace=*/0);
    auto assertFnTy = LLVM::LLVMFunctionType::get(voidTy, /*params=*/{ptrTy}, /*isVarArg=*/false);
    
    // Create assertion function for memory access failures
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("memory_assertion_failure")) {
      auto assertStub = builder.create<LLVM::LLVMFuncOp>(
          module.getLoc(), "memory_assertion_failure", assertFnTy);
      assertStub.setLinkage(LLVM::Linkage::External);
    }
    
    int unsafeMemoryOps = 0;
    
    // Check LLVM::LoadOp for tainted addresses
    module.walk([&](LLVM::LoadOp loadOp) {
      Value addr = loadOp.getAddr();
      
      // Check if the address depends on tainted data
      if (dependsOnTaintedData(addr)) {
        unsafeMemoryOps++;
        Location loc = loadOp.getLoc();
        
        std::string errorMsg = "Load " + std::to_string(unsafeMemoryOps) + 
                              " from address computed using dummy/tainted data - potential segfault";
        
        insertMemoryAssertion(builder, module, loc, errorMsg, unsafeMemoryOps, loadOp);
        
        llvm::errs() << "WARNING: Inserted assertion for potentially unsafe load at " 
                     << loc << "\n";
      }
    });
    
    // Check LLVM::StoreOp for tainted addresses  
    module.walk([&](LLVM::StoreOp storeOp) {
      Value addr = storeOp.getAddr();
      
      if (dependsOnTaintedData(addr)) {
        unsafeMemoryOps++;
        Location loc = storeOp.getLoc();
        
        std::string errorMsg = "Store " + std::to_string(unsafeMemoryOps) + 
                              " to address computed using dummy/tainted data - potential segfault";
        
        insertMemoryAssertion(builder, module, loc, errorMsg, unsafeMemoryOps, storeOp);
        
        llvm::errs() << "WARNING: Inserted assertion for potentially unsafe store at " 
                     << loc << "\n";
      }
    });
    
    // Check LLVM::GEPOp for tainted indices
    module.walk([&](LLVM::GEPOp gepOp) {
      Value basePtr = gepOp.getBase();
      auto indices = gepOp.getIndices();
      
      // Check for tainted base pointer
      bool hasTaintedBase = dependsOnTaintedData(basePtr);
      
      // Check each index for taint
      bool hasTaintedIndex = false;
      for (auto [i, index] : llvm::enumerate(indices)) {
        if (auto valueIndex = dyn_cast<Value>(index)) {
          if (dependsOnTaintedData(valueIndex)) {
            hasTaintedIndex = true;
            llvm::errs() << "WARNING: GEP using tainted index " << i << "\n";
            
            if (indices.size() > 1) {
              llvm::errs() << "  Multi-dimensional array access with tainted index!\n";
            }
          }
        }
      }
      
      if (hasTaintedBase || hasTaintedIndex) {
        unsafeMemoryOps++;
        Location loc = gepOp.getLoc();
        
        std::string errorMsg = "GEP " + std::to_string(unsafeMemoryOps);
        if (hasTaintedBase && hasTaintedIndex) {
          errorMsg += " with both tainted base pointer and indices - potential out-of-bounds access";
        } else if (hasTaintedBase) {
          errorMsg += " with tainted base pointer - potential invalid memory access";  
        } else {
          errorMsg += " with tainted indices - potential out-of-bounds access";
        }
        
        // Add extra warning for multi-dimensional access
        if (indices.size() > 1) {
          errorMsg += " (multi-dimensional array)";
        }
        
        insertMemoryAssertion(builder, module, loc, errorMsg, unsafeMemoryOps, gepOp);
        
        llvm::errs() << "WARNING: Inserted assertion for potentially unsafe GEP at " 
                    << loc << "\n";
      }
    });
    // Check memcpy/memmove operations
    module.walk([&](LLVM::CallOp callOp) {
      if (callOp.getCallee()) {
        StringRef calleeName = *callOp.getCallee();
        if (calleeName.contains("memcpy") || calleeName.contains("memmove") || 
            calleeName.contains("memset")) {
          
          // Check source and destination addresses
          auto operands = callOp.getOperands();
          if (operands.size() >= 2) {
            Value dest = operands[0];
            Value src = (operands.size() > 1) ? operands[1] : Value{};
            
            if (dependsOnTaintedData(dest) || (src && dependsOnTaintedData(src))) {
              unsafeMemoryOps++;
              Location loc = callOp.getLoc();
              
              std::string errorMsg = "Memory operation " + calleeName.str() + " " + 
                                    std::to_string(unsafeMemoryOps) + 
                                    " with tainted addresses - potential segfault";
              
              insertMemoryAssertion(builder, module, loc, errorMsg, unsafeMemoryOps, callOp);
              
              llvm::errs() << "WARNING: Inserted assertion for potentially unsafe " 
                           << calleeName << " at " << loc << "\n";
            }
          }
        }
      }
    });
    
    if (unsafeMemoryOps > 0) {
      llvm::errs() << "TAINT ANALYSIS: Found " << unsafeMemoryOps 
                   << " potentially unsafe memory operations\n";
    } else {
      llvm::errs() << "TAINT ANALYSIS: No unsafe memory operations found\n";
    }
  }

  // Helper method to insert memory assertion
  void insertMemoryAssertion(OpBuilder &builder, ModuleOp module, Location loc, 
                            const std::string &errorMsg, int errorId, Operation *beforeOp) {
    auto ptrTy = LLVM::LLVMPointerType::get(&getContext(), /*addrSpace=*/0);
    
    // Create global string for error message
    auto savedInsertionPoint = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(module.getBody());
    
    auto stringTy = LLVM::LLVMArrayType::get(builder.getI8Type(), errorMsg.length() + 1);
    auto errorGlobal = builder.create<LLVM::GlobalOp>(
        loc, stringTy, /*isConstant=*/true, LLVM::Linkage::Private,
        "memory_error_" + std::to_string(errorId),
        builder.getStringAttr(errorMsg + '\0'));
    
    builder.restoreInsertionPoint(savedInsertionPoint);
    
    // Insert assertion call before the potentially unsafe memory operation
    builder.setInsertionPoint(beforeOp);
    
    auto errorPtr = builder.create<LLVM::AddressOfOp>(loc, ptrTy, errorGlobal.getSymName());
    
    // Create assertion call
    builder.create<LLVM::CallOp>(
        loc, /*resultTypes=*/TypeRange{},
        /*callee=*/builder.getStringAttr("memory_assertion_failure"),
        /*args=*/ValueRange{errorPtr});
  }

void detectTaintedArrayAccess() {
  ModuleOp module = getOperation();
  
  // Look for patterns like: base_ptr + tainted_index * element_size
  module.walk([&](LLVM::GEPOp gepOp) {
    Value basePtr = gepOp.getBase();
    auto indices = gepOp.getIndices();
    
    // Check for tainted base pointer
    if (dependsOnTaintedData(basePtr)) {
      llvm::errs() << "WARNING: GEP using tainted base pointer\n";
    }
    
    // Check each index for taint
    for (auto [i, index] : llvm::enumerate(indices)) {
      if (auto valueIndex = dyn_cast<Value>(index)) {
        if (dependsOnTaintedData(valueIndex)) {
          llvm::errs() << "WARNING: GEP using tainted index " << i << "\n";
          
          // This is especially dangerous for multi-dimensional array access
          if (indices.size() > 1) {
            llvm::errs() << "  Multi-dimensional array access with tainted index!\n";
          }
        }
      }
    }
  });
}

public:
  void runOnOperation() override {
    llvm::errs() << "Running Taint Analysis on LLVM\n";
    
    // Step 0: Map block arguments to all their source values
    mapBlockArguments();

    // Step 1: Identify all tainted values
    identifyTaintedValues();
    
    // Step 2: Insert branch assertions
    insertBranchAssertions();
    
    // Step 3: Insert memory access assertions
    insertMemoryAccessAssertions();
    
    llvm::errs() << "Taint Analysis completed. Total tainted values: " 
                 << taintedValues.size() << "\n";
  }
};

} // namespace


namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTaintAnalysisOnLLVMPass() {
  return std::make_unique<TaintAnalysisOnLLVM>();
}

} // namespace triton
} // namespace mlir