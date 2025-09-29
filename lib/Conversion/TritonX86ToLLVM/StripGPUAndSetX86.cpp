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
#include <cstddef>
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton::nvgpu;
using ::mlir::triton::gpu::createDummyValue;

namespace {


static bool matchTritonAttr(StringRef name) {
  // Match any attribute that starts with "triton_gpu." or "nvvm."
  return  (
          name.rfind( "triton.") != StringRef::npos||
           name.rfind( "triton.metrics") != StringRef::npos||
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

    // 1) Collect any attributes named "triton_gpu.*" or "nvvm.*" and "triton_gpu.*" to rename
    SmallVector<StringRef> toErase;
    SmallVector<std::pair<StringRef, Attribute>> toRename;
    
    for (NamedAttribute na : module->getAttrs()) {
      StringRef name = na.getName().getValue();
      
      if (name.starts_with("triton_gpu.")) {
        // Rename triton_gpu.* to triton_x86.*
        std::string newName = name.str();
        newName.replace(0, 10, "triton_x86"); // Replace "triton_gpu" with "triton_x86"
        
        llvm::errs() << "Renaming module attribute: " << name << " -> " << newName << "\n";
        toRename.push_back({name, na.getValue()});
        toErase.push_back(name);
        
        // Add the renamed attribute
        module->setAttr(StringAttr::get(module.getContext(), newName), na.getValue());
      } else if (matchTritonAttr(name)) {
        llvm::errs() << "Erasing module attribute: " << name << "\n";
        toErase.push_back(name);
      }
    }

    // 2) Perform in-place edits on `module`.
    rewriter.startOpModification(op);
    
    // Remove old attributes
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
    auto loc = op->getLoc();
    int axis = 0;
    auto i32Ty = rewriter.getIntegerType(32);
    Value axisConst = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getIntegerAttr(i32Ty, axis));

    auto calleeName = rewriter.getStringAttr("nvvm_tid");
    auto call = rewriter.create<LLVM::CallOp>(
    op->getLoc(),
    /*resultTypes=*/TypeRange{i32Ty},
    /*callee=*/calleeName,
    /*args=*/ValueRange{axisConst});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};




struct ConvertReadTidY : public ConversionPattern{
  explicit ConvertReadTidY(MLIRContext *ctx)
      : ConversionPattern(NVVM::ThreadIdYOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    int axis = 1;
    auto i32Ty = rewriter.getIntegerType(32);
    Value axisConst = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getIntegerAttr(i32Ty, axis));

    auto calleeName = rewriter.getStringAttr("nvvm_tid");
    auto call = rewriter.create<LLVM::CallOp>(
    op->getLoc(),
    /*resultTypes=*/TypeRange{i32Ty},
    /*callee=*/calleeName,
    /*args=*/ValueRange{axisConst});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

struct ConvertReadTidZ : public ConversionPattern{
  explicit ConvertReadTidZ(MLIRContext *ctx)
      : ConversionPattern(NVVM::ThreadIdZOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    int axis = 2;
    auto i32Ty = rewriter.getIntegerType(32);
    Value axisConst = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getIntegerAttr(i32Ty, axis));

    auto calleeName = rewriter.getStringAttr("nvvm_tid");
    auto call = rewriter.create<LLVM::CallOp>(
    op->getLoc(),
    /*resultTypes=*/TypeRange{i32Ty},
    /*callee=*/calleeName,
    /*args=*/ValueRange{axisConst});
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

struct ConvertShflOp: public ConversionPattern{
  explicit ConvertShflOp(MLIRContext *ctx)
      : ConversionPattern(NVVM::ShflOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    
    // Create dummy results for operations that expect a result
    if (op->getNumResults() > 0) {
      SmallVector<Value> dummyResults;
      for (auto result : op->getResults()) {
        Value dummyResult = createDummyValue(rewriter, loc, result.getType(), 1);
        dummyResults.push_back(dummyResult);
      }
      rewriter.replaceOp(op, dummyResults);
    } else {
      rewriter.eraseOp(op);
    }
    
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
        && strRef.find("cp.") == std::string::npos
        && strRef.find("div.full.f32") == std::string::npos
        && strRef.find("prmt.b32") == std::string::npos
        && strRef.find("atom.") == std::string::npos
      ){
      return failure(); 
    }

    auto asmStr = asmOp.getAsmString();
    bool isLoad  = asmStr.contains("ld.global");
    bool isStore = asmStr.contains("st.global");
    bool isNCTAId = asmStr.contains("nctaid.");
    bool isCTAId = asmStr.contains("ctaid.") && !isNCTAId;
    bool isRemove = asmStr.contains("cp.") || asmStr.contains(".sync") || asmStr.contains(".async");
    bool isDivision = asmStr.contains("div.full.f32");
    bool isPrmt = asmStr.contains("prmt.b32");

    bool isLoadShared = asmStr.contains("ld.shared");
    bool isStoreShared = asmStr.contains("st.shared");



    Location loc = op->getLoc();
    
    llvm::errs() << "Converting InlineAsmOp: " << asmStr << "\n";

    /* ---------- SHARED MEMORY STORE ---------------------------------------- */
    if (isStoreShared) {
      llvm::errs() << "Converting st.shared: " << asmStr << "\n";
      
      if (asmOp.getNumOperands() != 3) return failure();
      if (asmOp.getNumResults() != 1) return failure();
      
      Type resTy = asmOp.getResult(0).getType();
      if (!mlir::isa<LLVM::LLVMVoidType>(resTy))
        return failure();
      
      Value ptr = asmOp.getOperand(0);   // $0 - destination pointer
      Value val = asmOp.getOperand(1);   // $1 - value to store
      Value pred = asmOp.getOperand(2);  // $2 - predicate
      
      if (pred) {
        Value oldVal = rewriter.create<LLVM::LoadOp>(loc, val.getType(), ptr);
        Value newVal = rewriter.create<LLVM::SelectOp>(loc, pred, val, oldVal);
        rewriter.create<LLVM::StoreOp>(loc, newVal, ptr);
      } else {
        rewriter.create<LLVM::StoreOp>(loc, val, ptr);
      }
      
      rewriter.eraseOp(op);
      return success();
    }
  
    /* ---------- PRMT.B32 BYTE PERMUTE ---------------------------------------- */
    if (isPrmt) {
      llvm::errs() << "Converting PTX prmt.b32: " << asmStr << "\n";
      
      if (asmOp.getNumOperands() != 1) return failure();
      if (asmOp.getNumResults() != 1) return failure();
      
      Value input = asmOp.getOperand(0);  // %11422 in your example
      Type resultType = asmOp.getResult(0).getType();
      
      // Verify it's the expected struct type
      auto structType = dyn_cast<LLVM::LLVMStructType>(resultType);
      if (!structType || structType.getBody().size() != 2) return failure();
      
      // For x86, we'll create dummy half-precision vectors
      // Since prmt.b32 is doing byte permutation to create f16 vectors,
      // we'll just create zero vectors as placeholders
      
      auto f16Ty = rewriter.getF16Type();
      auto vec2f16Ty = LLVM::getFixedVectorType(f16Ty, 2);
      
      // Create zero vectors for both outputs
      Value zeroF16 = rewriter.create<LLVM::ConstantOp>(
          loc, f16Ty, rewriter.getFloatAttr(f16Ty, 0.0));
      
      Value zeroVec1 = rewriter.create<LLVM::UndefOp>(loc, vec2f16Ty);
      zeroVec1 = rewriter.create<LLVM::InsertElementOp>(
          loc, zeroVec1, zeroF16, rewriter.create<LLVM::ConstantOp>(
              loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0)));
      zeroVec1 = rewriter.create<LLVM::InsertElementOp>(
          loc, zeroVec1, zeroF16, rewriter.create<LLVM::ConstantOp>(
              loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1)));
      
      Value zeroVec2 = rewriter.create<LLVM::UndefOp>(loc, vec2f16Ty);
      zeroVec2 = rewriter.create<LLVM::InsertElementOp>(
          loc, zeroVec2, zeroF16, rewriter.create<LLVM::ConstantOp>(
              loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0)));
      zeroVec2 = rewriter.create<LLVM::InsertElementOp>(
          loc, zeroVec2, zeroF16, rewriter.create<LLVM::ConstantOp>(
              loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1)));
      
      // Pack into the result struct
      Value result = rewriter.create<LLVM::UndefOp>(loc, structType);
      result = rewriter.create<LLVM::InsertValueOp>(
          loc, result, zeroVec1, rewriter.getDenseI64ArrayAttr({0}));
      result = rewriter.create<LLVM::InsertValueOp>(
          loc, result, zeroVec2, rewriter.getDenseI64ArrayAttr({1}));
      
      rewriter.replaceOp(op, result);
      return success();
    }
    
    if (isDivision) {
      llvm::errs() << "Converting PTX division: " << asmStr << "\n";
      
      // Expected: (f32, f32) -> f32 division
      if (asmOp.getNumOperands() != 2) return failure();
      if (asmOp.getNumResults() != 1) return failure();
      
      Value dividend = asmOp.getOperand(0);  // $1 
      Value divisor = asmOp.getOperand(1);   // $2 
      Type resultTy = asmOp.getResult(0).getType();
      
      if (!resultTy.isF32()) return failure();
      
      // Replace with regular LLVM division
      Value result = rewriter.create<LLVM::FDivOp>(loc, resultTy, dividend, divisor);
      rewriter.replaceOp(op, result);
      return success();
    }
    if (isRemove) {
      llvm::errs() << "Converting cp.* instruction: " << asmStr << "\n";
      
      // These are async memory copy operations - for x86, just ignore them
      // Since they return void, create a void result
      Type resTy = asmOp.getResult(0).getType();
      if (!mlir::isa<LLVM::LLVMVoidType>(resTy)) {
        return failure();
      }
      
      // Option 1: Replace with void undef (essentially a no-op)
      auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
      rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, voidTy);
      return success();
    }
   
    if (isLoad) {

      bool isVectorLoad = asmStr.contains("ld.global.v4") || asmStr.contains("ld.global.v2");
      if (isVectorLoad) {
        llvm::errs() << "Converting vector load: " << asmStr << "\n";
        
        // Expected: ptr, predicate -> struct<(i32, i32, i32, i32)>
        if (asmOp.getNumOperands() != 2) return failure();
        if (asmOp.getNumResults() != 1) return failure();
        
        Value ptr = asmOp.getOperand(0);
        Value pred = asmOp.getOperand(1);
        
        auto resultType = asmOp.getResult(0).getType();
        auto structType = dyn_cast<LLVM::LLVMStructType>(resultType);
        if (!structType) return failure();
        
        // Get the individual element types from the struct
        auto elementTypes = structType.getBody();
        if (elementTypes.size() != 4 && elementTypes.size() != 2 ) return failure(); // Expecting v4 or v2
        
        // Create individual loads for each element
        SmallVector<Value> loadedValues;
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 1); // addrspace 1
        
        for (size_t i = 0; i < elementTypes.size(); ++i) {
          // Calculate offset: ptr + i * sizeof(element)
          Value offset = rewriter.create<LLVM::ConstantOp>(
              loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(i * 4)); // 4 bytes per i32
          
          Value elemPtr = rewriter.create<LLVM::GEPOp>(
              loc, ptrType, rewriter.getI8Type(), ptr, ValueRange{offset});
          
          // Cast to the correct pointer type
          Value typedPtr = rewriter.create<LLVM::BitcastOp>(
              loc, LLVM::LLVMPointerType::get(rewriter.getContext(), 1), elemPtr);
          
          // Load the element
          Value loadedElem = rewriter.create<LLVM::LoadOp>(loc, elementTypes[i], typedPtr);
          
          // Apply predicate if needed
          if (pred) {
            Value zero = rewriter.create<LLVM::ConstantOp>(
                loc, elementTypes[i],
                rewriter.getIntegerAttr(cast<IntegerType>(elementTypes[i]), 0));
            loadedElem = rewriter.create<LLVM::SelectOp>(loc, elementTypes[i], pred, loadedElem, zero);
          }
          
          loadedValues.push_back(loadedElem);
        }
        
        // Pack into struct
        Value result = rewriter.create<LLVM::UndefOp>(loc, structType);
        for (size_t i = 0; i < loadedValues.size(); ++i) {
          result = rewriter.create<LLVM::InsertValueOp>(
              loc, result, loadedValues[i], rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(i)}));
        }
        
        rewriter.replaceOp(op, result);
        return success();
      }

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

    /* ---------- GLOBAL STORE ---------------------------------------- */
    if(isStore){
      bool isVectorStoreV4 = asmStr.contains("st.global.v4");
      bool isVectorStoreV2 = asmStr.contains("st.global.v2");
      
      if (isVectorStoreV2) {
        llvm::errs() << "Converting vector store: " << asmStr << " not supported \n";
        return failure();
      }

      if (isVectorStoreV4) {
        llvm::errs() << "Converting vector store: " << asmStr << "\n";
        
        // Expected: val0, val1, val2, val3, ptr, predicate -> void
        if (asmOp.getNumOperands() != 6) return failure(); // 4 values + ptr + predicate
        if (asmOp.getNumResults() != 1) return failure();
        
        Type resTy = asmOp.getResult(0).getType();
        if (!mlir::isa<LLVM::LLVMVoidType>(resTy))
          return failure();
        
        // Extract operands
        Value val0 = asmOp.getOperand(0);
        Value val1 = asmOp.getOperand(1);
        Value val2 = asmOp.getOperand(2);
        Value val3 = asmOp.getOperand(3);
        Value ptr = asmOp.getOperand(4);
        Value pred = asmOp.getOperand(5);
        
        SmallVector<Value> values = {val0, val1, val2, val3};
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 1); // addrspace 1
        
        // Store each element individually
        for (size_t i = 0; i < values.size(); ++i) {
          // Calculate offset: ptr + i * sizeof(element)
          Value offset = rewriter.create<LLVM::ConstantOp>(
              loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(i * 4)); // 4 bytes per i32
          
          Value elemPtr = rewriter.create<LLVM::GEPOp>(
              loc, ptrType, rewriter.getI8Type(), ptr, ValueRange{offset});
          
          // Cast to the correct pointer type
          Value typedPtr = rewriter.create<LLVM::BitcastOp>(
              loc, LLVM::LLVMPointerType::get(rewriter.getContext(), 1), elemPtr);
          
          // Apply predicate to the value if needed
          Value dataToStore = values[i];
          if (pred) {
            Value old = rewriter.create<LLVM::LoadOp>(loc, values[i].getType(), typedPtr);
            dataToStore = rewriter.create<LLVM::SelectOp>(loc, values[i].getType(), pred, values[i], old);
          }
          
          // Store the element
          rewriter.create<LLVM::StoreOp>(loc, dataToStore, typedPtr);
        }
        
        rewriter.eraseOp(op);
        // auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
        // rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, voidTy);
        return success();
      }

      // llvm::errs() << "Converting st.global : " << asmOp.getAsmString() << " (" << asmOp.getNumOperands() << ") -> #of results (" << asmOp.getNumResults() << "\n";
      if(asmOp.getNumResults() != 1) {
        return failure();
      }
      Type resTy = asmOp.getResult(0).getType();
      if (!mlir::isa<LLVM::LLVMVoidType>(resTy))
        return failure();
      
      if (asmOp.getNumOperands() != 2 &&
          asmOp.getNumOperands() != 3){
        return failure();
      }              

      Value val  = asmOp.getOperand(0);
      Value ptr  = asmOp.getOperand(1);
      Value pred = (asmOp.getNumOperands() == 3) ? asmOp.getOperand(2) : Value();

      Value data = val;
      if (pred) {
        Value old = rewriter.create<LLVM::LoadOp>(loc, val.getType(), ptr);
        data = rewriter.create<LLVM::SelectOp>(loc, val.getType(), pred, val, old);
      }

      rewriter.create<LLVM::StoreOp>(loc, data, ptr);
      rewriter.eraseOp(op);
      return success();
    }

    /* ---------- CTA ID ---------------------------------------- */
    if (isCTAId){
      int axis = -1;
      bool isCtaIdX = asmStr.contains("%ctaid.x");
      bool isCtaIdY = asmStr.contains("%ctaid.y");
      bool isCtaIdZ = asmStr.contains("%ctaid.z");
      if(isCtaIdX) axis = 0;
      else if(isCtaIdY) axis = 1;
      else if(isCtaIdZ) axis = 2;
      
      if (asmOp.getNumOperands() != 0) return failure();
      if (asmOp.getNumResults()  != 1) return failure();
      Type resTy = asmOp.getResult(0).getType();
      if (!resTy.isSignlessInteger(32)) return failure();

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

    if (isNCTAId){
      int axis = -1;
      bool isCtaIdX = asmStr.contains("%nctaid.x");
      bool isCtaIdY = asmStr.contains("%nctaid.y");
      bool isCtaIdZ = asmStr.contains("%nctaid.z");
      if(isCtaIdX) axis = 0;
      else if(isCtaIdY) axis = 1;
      else if(isCtaIdZ) axis = 2;
      
      if (asmOp.getNumOperands() != 0) return failure();
      if (asmOp.getNumResults()  != 1) return failure();
      Type resTy = asmOp.getResult(0).getType();
      if (!resTy.isSignlessInteger(32)) return failure();

      auto i32Ty = rewriter.getIntegerType(32);
      Value axisConst = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getIntegerAttr(i32Ty, axis));

      auto call = rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          op,
          /*resultTypes=*/TypeRange{i32Ty},
          /*callee=*/rewriter.getStringAttr("nvvm_nctaid"),
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
    std::string attrName = "unknown";
    bool hasTritonAttr = false;
    for (NamedAttribute na : allocaOp->getAttrs()) {
      StringRef name = na.getName().getValue();
      if (matchTritonAttr(name)) {
        hasTritonAttr = true;
        attrName = name.str();
        break;
      }
    }

    if (!hasTritonAttr)
      return failure();

    Location loc = op->getLoc();
    
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module) {
      return failure();
    }

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
    
    static int counter = 0;
    std::string symbolName = "metrics_name_" + std::to_string(counter++); 
    auto arrayTy = LLVM::LLVMArrayType::get(rewriter.getI8Type(), attrName.length() + 1);
    
        auto savedInsertionPoint = rewriter.saveInsertionPoint();
    
    // Set insertion point to module level
    rewriter.setInsertionPointToStart(module.getBody());
    
    auto nameConstant = rewriter.create<LLVM::GlobalOp>(
        loc, 
        arrayTy,
        /*isConstant=*/true,
        LLVM::Linkage::Private,
        symbolName,
        rewriter.getStringAttr(attrName + '\0'));
    
    // Restore insertion point
    rewriter.restoreInsertionPoint(savedInsertionPoint);
    
    // Create string constant for the attribute name
    auto stringTy = LLVM::LLVMPointerType::get(rewriter.getContext(), /*addrSpace=*/0);
    
    auto namePtr = rewriter.create<LLVM::AddressOfOp>(loc, stringTy, nameConstant.getSymName());
    auto nameGEP = rewriter.create<LLVM::GEPOp>(
    loc, stringTy, arrayTy, namePtr,  // Use arrayTy instead of rewriter.getI8Type()
    ValueRange{rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getIntegerAttr(rewriter.getI64Type(), 0)),
               rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getIntegerAttr(rewriter.getI64Type(), 0))});
    // Call metrics_alloca(size, name) -> ptr
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), /*addrSpace=*/0);
    auto call = rewriter.create<LLVM::CallOp>(
        loc,
        /*resultTypes=*/TypeRange{ptrTy},
        /*callee=*/rewriter.getStringAttr("metrics_alloca"),
        /*args=*/ValueRange{totalSize, nameGEP});

    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

struct ConvertUnrealizedConversionCast : public ConversionPattern {
  explicit ConvertUnrealizedConversionCast(MLIRContext *ctx)
      : ConversionPattern(UnrealizedConversionCastOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto castOp = dyn_cast<UnrealizedConversionCastOp>(op);
    if (!castOp)
      return failure();

    if (castOp.getInputs().size() != 1 || castOp.getOutputs().size() != 1)
      return failure();

    Value input = castOp.getInputs()[0];
    Value output = castOp.getOutputs()[0];
    Type inputType = input.getType();
    Type outputType = output.getType();

    Location loc = castOp.getLoc();

    llvm::errs() << "Converting unrealized_conversion_cast from " 
                 << inputType << " to " << outputType << "\n";

    // CRITICAL: Block any conversion TO Triton types
    if (
        mlir::isa<RankedTensorType>(outputType) ||
        outputType.getDialect().getNamespace() == "tt" ||
        outputType.getDialect().getNamespace() == "triton_gpu") {
      
      llvm::errs() << "BLOCKING conversion to Triton type: " << outputType << "\n";
      rewriter.replaceOp(castOp, input);
      return success();

      if (auto tritonPtrType = dyn_cast<triton::PointerType>(outputType)) {
        // Create equivalent LLVM pointer type
        auto llvmPtrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
        
        if (inputType == llvmPtrType) {
          // Direct replacement - don't create tt.ptr
          rewriter.replaceOp(castOp, input);
        } else {
          // Convert to LLVM pointer, not tt.ptr
          Value converted;
          if (auto inputPtrType = dyn_cast<LLVM::LLVMPointerType>(inputType)) {
            // Pointer to pointer cast
            converted = rewriter.create<LLVM::BitcastOp>(loc, llvmPtrType, input);
          } else if (auto inputIntType = dyn_cast<IntegerType>(inputType)) {
            // Int to pointer
            converted = rewriter.create<LLVM::IntToPtrOp>(loc, llvmPtrType, input);
          } else {
            // Fallback bitcast
            converted = rewriter.create<LLVM::BitcastOp>(loc, llvmPtrType, input);
          }
          rewriter.replaceOp(castOp, converted);
        }
        return success();
      }
      
      // For other Triton types, force replacement with input
      llvm::errs() << "Forcing replacement with input to prevent illegal operation\n";
      rewriter.replaceOp(castOp, input);
      return success();
    }

    if (
        mlir::isa<RankedTensorType>(inputType) ||
        inputType.getDialect().getNamespace() == "tt" ||
        inputType.getDialect().getNamespace() == "triton_gpu") {
      
      llvm::errs() << "BLOCKING conversion from Triton type: " << inputType << "\n";
      rewriter.replaceOp(castOp, input);
      return success();
    }

    // Rest of your existing logic for LLVM-to-LLVM conversions...
    
    // Identical types
    if (inputType == outputType) {
      rewriter.replaceOp(castOp, input);
      return success();
    }

    // LLVM pointer conversions
    auto inputPtrTy = dyn_cast<LLVM::LLVMPointerType>(inputType);
    auto outputPtrTy = dyn_cast<LLVM::LLVMPointerType>(outputType);
    
    if (inputPtrTy && outputPtrTy) {
      if (inputPtrTy.getAddressSpace() != outputPtrTy.getAddressSpace()) {
        Value converted = rewriter.create<LLVM::AddrSpaceCastOp>(
            loc, outputPtrTy, input);
        rewriter.replaceOp(castOp, converted);
        return success();
      }
      rewriter.replaceOp(castOp, input);
      return success();
    }

    // Integer conversions
    auto inputIntTy = dyn_cast<IntegerType>(inputType);
    auto outputIntTy = dyn_cast<IntegerType>(outputType);
    
    if (inputIntTy && outputIntTy) {
      if (inputIntTy.getWidth() < outputIntTy.getWidth()) {
        Value converted = rewriter.create<LLVM::ZExtOp>(loc, outputType, input);
        rewriter.replaceOp(castOp, converted);
        return success();
      } else if (inputIntTy.getWidth() > outputIntTy.getWidth()) {
        Value converted = rewriter.create<LLVM::TruncOp>(loc, outputType, input);
        rewriter.replaceOp(castOp, converted);
        return success();
      }
      rewriter.replaceOp(castOp, input);
      return success();
    }

    // Int to pointer
    if (inputIntTy && outputPtrTy) {
      Value converted = rewriter.create<LLVM::IntToPtrOp>(loc, outputType, input);
      rewriter.replaceOp(castOp, converted);
      return success();
    }

    // Pointer to int
    if (inputPtrTy && outputIntTy) {
      Value converted = rewriter.create<LLVM::PtrToIntOp>(loc, outputType, input);
      rewriter.replaceOp(castOp, converted);
      return success();
    }

    // Only allow LLVM-to-LLVM bitcast
    if (inputType.getDialect().getNamespace() == "llvm" &&
        outputType.getDialect().getNamespace() == "llvm") {
      Value converted = rewriter.create<LLVM::BitcastOp>(loc, outputType, input);
      rewriter.replaceOp(castOp, converted);
      return success();
    }

    llvm::errs() << "ERROR: Cannot convert between incompatible types:\n";
    llvm::errs() << "  From: " << inputType << "\n";
    llvm::errs() << "  To: " << outputType << "\n";
    
    return failure(); // Fail conversion for truly incompatible types
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
  patterns.add<ConvertReadTidY>(ctx);
  patterns.add<ConvertReadTidZ>(ctx);
  patterns.add<ConvertClusterId>(ctx);
  patterns.add<ConvertBarrier0Op>(ctx);
  patterns.add<ConvertShflOp>(ctx);
  patterns.add<ConvertGlobalInline>(ctx);
  patterns.add<ConvertMetricsAlloca>(ctx);
}

void mlir::triton::populateStripGPUAndSetX86CleanUp(
    LLVMTypeConverter & typeConverter,
    RewritePatternSet &patterns,
    const TargetInfoBase & targetInfo) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<ConvertUnrealizedConversionCast>(ctx);
}
