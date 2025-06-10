//===- StripGPUAndSetX86.cpp -----------------------------------*- C++ -*-===//
//
// Strip all Triton/NVVM GPU attributes from a module and its LLVM functions,
// then inject an x86_64 triple + data layout.
//
// These two ConversionPatterns get pulled into Triton’s existing
// GPU→LLVM conversion pipeline. Whenever we mutate the matched operation in place,
// we call `rewriter.notifyRootChanged(op)` (defined in RewriterBase).
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"  // for ConversionPattern, etc.
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

using namespace mlir;
using namespace mlir::triton::nvgpu;

namespace {


static bool matchTritonAttr(StringRef name) {
  // Match any attribute that starts with "triton_gpu." or "nvvm."
  return  (
          name.rfind( "triton.") != StringRef::npos||
          name.rfind( "triton_gpu.") != StringRef::npos ||
          name.rfind("nvvm.") != StringRef::npos || 
          name.rfind("tt.") != StringRef::npos);
}


///----------------------------------------------------------------------
/// ConversionPattern #1: ModuleOp
///
///   • Erase any module-level attributes whose name starts with "triton_gpu."
///     or "nvvm."  
///   • Overwrite (or set) the module’s `llvm.triple` and `llvm.data_layout`
///     to x86_64‐Linux values.  
///
///   After performing these in-place edits, we immediately call
///   `rewriter.notifyRootChanged(module)` so that MLIR’s DialectConversion
///   framework knows we did modify the root operation.
///----------------------------------------------------------------------
struct StripGPUAttrsInModule : public ConversionPattern {
  explicit StripGPUAttrsInModule(MLIRContext *ctx)
      : ConversionPattern(ModuleOp::getOperationName(), /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = cast<ModuleOp>(op);

    // 1) Collect any attributes named "triton_gpu.*" or "nvvm.*".
    SmallVector<StringRef> toErase;
    for (NamedAttribute na : module->getAttrs()) {
      StringRef name = na.getName().getValue();
      if (matchTritonAttr(name)) {
        llvm::errs() << "Erasing module attribute: " << name << "\n";
        toErase.push_back(name);
      }
    }

    // 2) Perform in-place edits on `module`.
    rewriter.startOpModification(op);
    for (StringRef n : toErase)
      module->removeAttr(n);

    module->setAttr("llvm.triple",
                    StringAttr::get(module.getContext(),
                                    "x86_64-pc-linux-gnu"));
    module->setAttr("llvm.data_layout",
                    StringAttr::get(module.getContext(),
                                    "e-m:e-i64:64-f80:128-n8:16:32:64-S128"));

    // 3) Notify MLIR that `module` was modified in place.
    rewriter.finalizeOpModification(module);
    return success();
  }
};

///----------------------------------------------------------------------
/// ConversionPattern #2: LLVMFuncOp
///
///   • Drop any function-level NVVM/Triton attributes such as "nvvm.kernel",
///     "nvvm.maxntid", "triton_gpu.localloadop.metric", etc.  
///   • Drop any leftover attributes whose name starts with "triton_gpu." or
///     "nvvm.".  
///   • If a parameter type is `!llvm.ptr<X>` with X≠0, rebuild the function type
///     so that those pointers become addrspace=0 (i.e. `!llvm.ptr<0>`).  
///
///   After these in-place edits, call `rewriter.notifyRootChanged(func)`.
///----------------------------------------------------------------------
struct StripGPUAttrsInFunc : public ConversionPattern {
  explicit StripGPUAttrsInFunc(MLIRContext *ctx)
      : ConversionPattern(LLVM::LLVMFuncOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto func = cast<LLVM::LLVMFuncOp>(op);
    bool needUpdate = false;


    SmallVector<StringRef> toErase;
    for (NamedAttribute na : func->getAttrs()) {
      StringRef name = na.getName().getValue();
      if (matchTritonAttr(name))
        toErase.push_back(name);
    }

    needUpdate = !toErase.empty();

    auto fnTy  = func.getFunctionType();
    bool rewriteTy = false;
    SmallVector<Type> newParams;
    newParams.reserve(fnTy.getNumParams());

    for (Type t : fnTy.getParams()) {
      if (auto ptrTy = mlir::dyn_cast<LLVM::LLVMPointerType>(t)) {
        if (ptrTy.getAddressSpace() != 0) {
          // Replace with an opaque pointer in addrspace=0.
          newParams.push_back(
              LLVM::LLVMPointerType::get(func.getContext(), /*addrSpace=*/0));
          rewriteTy = true;
        } else {
          newParams.push_back(t);
        }
      } else {
        newParams.push_back(t);
      }
    }
    if (rewriteTy)
      needUpdate = true;

    // If nothing changed, return failure so that other patterns can run.
    if (!needUpdate)
      return failure();

    rewriter.startOpModification(op);
  
    if (!toErase.empty()) {
      for (auto &n : toErase){
        llvm::errs() << "Erasing llvm func attribute: " << n << "\n";
        func->removeAttr(n);
      }
    }

    if (rewriteTy) {
      // Build and install the new function type
      auto newFnTy = LLVM::LLVMFunctionType::get(
          fnTy.getReturnType(), newParams, fnTy.isVarArg());
      func.setType(newFnTy);

      // Update the entry‐block arguments to match the new types
      Block &entry = func.getBody().front();
      for (unsigned i = 0, e = newParams.size(); i != e; ++i)
        entry.getArgument(i).setType(newParams[i]);
     }

    rewriter.finalizeOpModification(op);
    return success();
  }
};

struct ConvertReadTidX : public ConversionPattern{
  explicit ConvertReadTidX(MLIRContext *ctx)
      : ConversionPattern(NVVM::ThreadIdXOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
     // Create a call to get_tid_x(): i32 = call @get_tid_x()
    auto i32Ty = rewriter.getI32Type();
    auto calleeName = rewriter.getStringAttr("nvvm_tid_x");
    auto call = rewriter.create<LLVM::CallOp>(
    op->getLoc(),
    /*resultTypes=*/TypeRange{i32Ty},
    /*callee=*/calleeName,
    /*args=*/ValueRange{});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};


struct ConvertBarrier0Op: public ConversionPattern{
  explicit ConvertBarrier0Op(MLIRContext *ctx)
      : ConversionPattern(NVVM::Barrier0Op::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto calleeName = rewriter.getStringAttr("nvvm_barrier0");
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op,
        /*resultTypes=*/TypeRange{},   // void
        /*callee=*/calleeName,
        /*args=*/ValueRange{});        // no operands
    return success();
  }
};

struct ConvertClusterId : public ConversionPattern{
  explicit ConvertClusterId(MLIRContext *ctx)
      : ConversionPattern(nvgpu::ClusterCTAIdOp::getOperationName(),
                          /*benefit=*/1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
     // Create a call to get_tid_x(): i32 = call @get_tid_x()
    auto i32Ty = rewriter.getI32Type();
    auto calleeName = rewriter.getStringAttr("nvgpu_cluster_id");
    auto call = rewriter.create<LLVM::CallOp>(
    op->getLoc(),
    /*resultTypes=*/TypeRange{i32Ty},
    /*callee=*/calleeName,
    /*args=*/ValueRange{});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Match any LLVM::InlineAsmOp whose asm string contains "ld.global".
// struct ConvertGlobalInlineLoad : public ConversionPattern {
//   explicit ConvertGlobalInlineLoad(MLIRContext *ctx)
//       : ConversionPattern(LLVM::InlineAsmOp::getOperationName(), 1, ctx) {}

//   LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
//                                 ConversionPatternRewriter &rewriter) const override {
//     auto asmOp = cast<LLVM::InlineAsmOp>(op);

//     if (asmOp.getAsmString().str().find("ld.global") == std::string::npos){
//       return failure();
//     }
//     // Expected signature: (ptr, i1) -> T   (T is i8/i16/i32/i64)
//     if (asmOp.getNumOperands() != 2 || asmOp.getNumResults() != 1)
//       return failure();
//     Value  ptr  = asmOp.getOperand(0);
//     Value  pred = asmOp.getOperand(1);
//     Type   elemTy = asmOp.getResultTypes().front();

//     // 1) Plain LLVM load  (unconditional)
//     auto loaded = rewriter.create<LLVM::LoadOp>(op->getLoc(), elemTy, ptr);

//     // 2) Zero literal of the same element type
//     Value zero;
//     if (elemTy.isInteger(8) || elemTy.isInteger(16) ||
//         elemTy.isInteger(32) || elemTy.isInteger(64))
//       zero = rewriter.create<LLVM::ConstantOp>(
//           op->getLoc(), elemTy, rewriter.getIntegerAttr(elemTy, 0));
//     else
//       return failure();   // unsupported element size

//     // 3) `select pred, loaded, zero`
//     auto selected = rewriter.create<LLVM::SelectOp>(
//         op->getLoc(), elemTy, pred, loaded, zero);

//     // Replace the asm op with the selected value
//     rewriter.replaceOp(op, selected.getResult());
//     return success();
//   }
// };

/// Replaces scalar `ld.global` inline-asm with LLVM load (and select).
///
/// Accepts either   (ptr)       -> i8/i16/i32/i64   (un-masked)
///            or    (ptr, i1)   -> i8/i16/i32/i64   (masked)
///
/// After this pattern runs there are *no* llvm.inline_asm ops of that shape.
// struct ConvertGlobalInline : public mlir::ConversionPattern {
//   explicit ConvertGlobalInline(MLIRContext *ctx)
//       : ConversionPattern(LLVM::InlineAsmOp::getOperationName(),
//                           /*benefit=*/1, ctx) {}

//   LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
//                                 ConversionPatternRewriter &rewriter) const override {
//     auto asmOp = dyn_cast<LLVM::InlineAsmOp>(op);
//     if (!asmOp)
//       return failure();

//     // one scalar integer result
//     if (asmOp.getNumResults() != 1)
//       return failure();
//     Type elemTy = asmOp.getResultTypes().front();
//     if (!elemTy.isIntOrIndex())
//       return failure();

//     // (ptr)  or  (ptr , i1)
//     if (asmOp.getNumOperands() != 1 && asmOp.getNumOperands() != 2)
//       return failure();
//     Value ptr  = asmOp.getOperand(0);
//     if (!mlir::isa<LLVM::LLVMPointerType>(ptr.getType()))
//       return failure();
//     bool masked = (asmOp.getNumOperands() == 2);
//     Value pred  = masked ? asmOp.getOperand(1) : Value();

//     Location loc = op->getLoc();

//     auto loaded = rewriter.create<LLVM::LoadOp>(loc, elemTy, ptr);

//     Value replacement = loaded.getResult();
//     if (masked) {
//       auto zero = rewriter.create<LLVM::ConstantOp>(
//           loc, elemTy,
//           rewriter.getIntegerAttr(mlir::cast<IntegerType>(elemTy), 0));

//       // select %pred, %ld, 0
//       replacement = rewriter.create<LLVM::SelectOp>(
//           loc, elemTy, pred, loaded, zero);
//     }

//     rewriter.replaceOp(op, replacement);
//     return success();
//   }
// };
//===----------------------------------------------------------------------===//
//  Convert scalar ld.global / st.global inline-asm
//  after Triton lowering (masked + unmasked)
//===----------------------------------------------------------------------===//

struct ConvertGlobalInline : public ConversionPattern{
  explicit ConvertGlobalInline(MLIRContext *ctx)
      : ConversionPattern(LLVM::InlineAsmOp::getOperationName(),
                          1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value>,
                                ConversionPatternRewriter &rewriter) const override {

    auto asmOp = llvm::dyn_cast<LLVM::InlineAsmOp>(op);
    if (!asmOp) return failure();

    auto strRef = asmOp.getAsmString().str();
    if (strRef.find("ld.global") == std::string::npos
        && strRef.find("st.global") == std::string::npos
        && strRef.find("st.shared") == std::string::npos
        && strRef.find("ld.shared") == std::string::npos
        && strRef.find("ctaid.") == std::string::npos
      ){
      return failure(); 
    }

    /* ---------- fast detect load vs store --------------------------------- */
    // Triton keeps the original constraint string
    //   LOAD  : "=c,l,b"   (one output “=c”, two inputs “l,b”)
    //   STORE : "r,l,b"    (zero outputs, three inputs)
    auto asmStr = asmOp.getAsmString();
    bool isLoad  = asmStr.contains("ld.global");
    bool isStore = asmStr.contains("st.global");
    bool isCTAId = asmStr.contains("ctaid.");
    if (!isLoad && !isStore && !isCTAId) return failure();

    Location loc = op->getLoc();

    /* ---------- scalar GLOBAL LOAD ---------------------------------------- */
    if (isLoad) {
      // ptr [, pred]  -> iN
      if (asmOp.getNumOperands() != 1 && asmOp.getNumOperands() != 2)
        return failure();
      if (asmOp.getNumResults() != 1)
        return failure();

      Value ptr  = asmOp.getOperand(0);
      Value pred = (asmOp.getNumOperands() == 2) ? asmOp.getOperand(1) : Value();
      Type elemTy = asmOp.getResult(0).getType();
      if (!elemTy.isIntOrIndex()) return failure();

      // unconditional load
      Value ld = rewriter.create<LLVM::LoadOp>(loc, elemTy, ptr);
      Value repl = ld;

      if (pred) {
        Value zero = rewriter.create<LLVM::ConstantOp>(
            loc, elemTy,
            rewriter.getIntegerAttr(mlir::cast<mlir::IntegerType>(elemTy), 0));
        repl = rewriter.create<LLVM::SelectOp>(loc, elemTy, pred, ld, zero);
      }
      rewriter.replaceOp(op, repl);
      return success();
    }
  
    if(isStore){
      // llvm::errs() << "Converting st.global : " << asmOp.getAsmString() << " (" << asmOp.getNumOperands() << ") -> #of results (" << asmOp.getNumResults() << "\n";
      if(asmOp.getNumResults() != 1) {
        return failure();
      }
      Type resTy = asmOp.getResult(0).getType();
      if (!mlir::isa<LLVM::LLVMVoidType>(resTy))
        return failure();
      // (val, ptr [, pred])  →  void
      if (asmOp.getNumOperands() != 2 &&
          asmOp.getNumOperands() != 3){
        return failure();
      }              

      Value val  = asmOp.getOperand(0);
      Value ptr  = asmOp.getOperand(1);
      Value pred = (asmOp.getNumOperands() == 3) ? asmOp.getOperand(2) : Value();

      Value data = val;
      if (pred) {
        // masked store: keep old value where predicate is false
        Value old = rewriter.create<LLVM::LoadOp>(loc, val.getType(), ptr);
        data = rewriter.create<LLVM::SelectOp>(loc, val.getType(), pred, val, old);
      }

      rewriter.create<LLVM::StoreOp>(loc, data, ptr);
      rewriter.eraseOp(op);
      return success();
    }

    if (isCTAId){

      int axis = -1;
      bool isCtaIdX = asmStr.contains("%ctaid.x");
      bool isCtaIdY = asmStr.contains("%ctaid.y");
      bool isCtaIdZ = asmStr.contains("%ctaid.z");
      if(isCtaIdX) axis = 0;
      else if(isCtaIdY) axis = 1;
      else if(isCtaIdZ) axis = 2;
      
      llvm::errs() << "Converting ctaid: " << asmOp.getAsmString() << " " << asmOp.getNumOperands() << " -> #of results (" << asmOp.getNumResults() << "\n";
      
      // Expect no operands, one i32 result
      if (asmOp.getNumOperands() != 0)          return failure();
      if (asmOp.getNumResults()  != 1)          return failure();
      Type resTy = asmOp.getResult(0).getType();
      if (!resTy.isSignlessInteger(32))   return failure();

      auto i32Ty = rewriter.getIntegerType(32);
      Value axisConst = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getIntegerAttr(i32Ty, axis));

      auto call = rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          op,
          /*resultTypes=*/TypeRange{i32Ty},
          /*callee=*/rewriter.getStringAttr("nvvm_ctaid"),
          /*args=*/ValueRange{axisConst});

      return success();
      
    }

    return failure();
  }
};


struct ConvertMetricsAlloca : public ConversionPattern {
  explicit ConvertMetricsAlloca(MLIRContext *ctx)
      : ConversionPattern(LLVM::AllocaOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto allocaOp = dyn_cast<LLVM::AllocaOp>(op);
    if (!allocaOp) 
      return failure();

    // Check if it has any triton.* attributes
    bool hasTritonAttr = false;
    for (NamedAttribute na : allocaOp->getAttrs()) {
      StringRef name = na.getName().getValue();
      if (matchTritonAttr(name)) {
        hasTritonAttr = true;
        break;
      }
    }

    if (!hasTritonAttr)
      return failure();

    Location loc = op->getLoc();
    
    // Get the allocation size in bytes
    Value arraySize = allocaOp.getArraySize();
    Type elemType = allocaOp.getElemType();
    
    // Calculate total size = arraySize * sizeof(elemType)
    auto i64Ty = rewriter.getI64Type();
    Value elemSizeBytes;
    
    if (auto intTy = dyn_cast<IntegerType>(elemType)) {
      int64_t bits = intTy.getWidth();
      int64_t bytes = (bits + 7) / 8; // Round up to nearest byte
      elemSizeBytes = rewriter.create<LLVM::ConstantOp>(
          loc, i64Ty, rewriter.getIntegerAttr(i64Ty, bytes));
    } else if (elemType.isF32()) {
      elemSizeBytes = rewriter.create<LLVM::ConstantOp>(
          loc, i64Ty, rewriter.getIntegerAttr(i64Ty, 4));
    } else if (elemType.isF64()) {
      elemSizeBytes = rewriter.create<LLVM::ConstantOp>(
          loc, i64Ty, rewriter.getIntegerAttr(i64Ty, 8));
    } else {
      return failure();
    }

    Value arraySizeI64 = arraySize;
    if (arraySize.getType() != i64Ty) {
      arraySizeI64 = rewriter.create<LLVM::ZExtOp>(loc, i64Ty, arraySize);
    }

    // totalSize = arraySize * elemSizeBytes
    Value totalSize = rewriter.create<LLVM::MulOp>(loc, i64Ty, arraySizeI64, elemSizeBytes);

    // Call metrics_alloca(size) -> ptr
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), /*addrSpace=*/0);
    auto call = rewriter.create<LLVM::CallOp>(
        loc,
        /*resultTypes=*/TypeRange{ptrTy},
        /*callee=*/rewriter.getStringAttr("metrics_alloca"),
        /*args=*/ValueRange{totalSize});

    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};


} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Public hook: Triton calls this to add our patterns to its pattern set.
//===----------------------------------------------------------------------===//
void mlir::triton::populateStripGPUAndSetX86(
    LLVMTypeConverter & typeConverter,
    RewritePatternSet &patterns,
    const TargetInfoBase & targetInfo) {
  MLIRContext *ctx = patterns.getContext();

  patterns.add<StripGPUAttrsInModule>(ctx);
  patterns.add<StripGPUAttrsInFunc>(ctx);
  patterns.add<ConvertReadTidX>(ctx);
  patterns.add<ConvertClusterId>(ctx);
  patterns.add<ConvertBarrier0Op>(ctx);
  patterns.add<ConvertGlobalInline>(ctx);
  patterns.add<ConvertMetricsAlloca>(ctx);
}
