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
#define GEN_PASS_DEF_CONVERTTRITONGPUTOLLVM
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::NVIDIA;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

// pass ws related named attrs.
static void addAttrs(Operation *op, ArrayRef<mlir::NamedAttribute> attrs) {
  for (const NamedAttribute attr : attrs)
    op->setAttr(attr.getName(), attr.getValue());
}

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<index::IndexDialect>();
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

static bool matchTritonAttr(StringRef name) {
  // Match any attribute that starts with "triton_gpu." or "nvvm."
  return  (
          name.rfind( "triton.") != StringRef::npos||
          name.rfind( "triton_gpu.") != StringRef::npos ||
          name.rfind("nvvm.") != StringRef::npos || 
          name.rfind("tt.") != StringRef::npos);
}

class X86LLVMConversionTarget : public ConversionTarget {
public:
  explicit X86LLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    // addLegalDialect<NVVM::NVVMDialect>();
    // addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();

    addIllegalDialect<NVVM::NVVMDialect>();
    addIllegalDialect<mlir::triton::nvgpu::NVGPUDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    
    addDynamicallyLegalOp<LLVM::InlineAsmOp>(
      [&](LLVM::InlineAsmOp op) {
        auto strRef = op.getAsmString().str();
        if (strRef.find("ld.global") != std::string::npos
            || strRef.find("st.global") != std::string::npos
            || strRef.find("st.shared") != std::string::npos
            || strRef.find("ld.shared") != std::string::npos
            || strRef.find("ctaid.") != std::string::npos
            || strRef.find("cp.") != std::string::npos
            || strRef.find("div.full.f32") != std::string::npos
          ){
          return false;
        }
        return true;
    });

    addDynamicallyLegalOp<ModuleOp>([&](ModuleOp module) {
      for (auto &attr : module->getAttrs()) {
        StringRef name = attr.getName().getValue();
        if (matchTritonAttr(name)) {
          llvm::errs() << "Module illegal due to attribute: " << name << "\n";
          return false;
        }
      }
      return true;
    });

    addDynamicallyLegalOp<LLVM::LLVMFuncOp>([](LLVM::LLVMFuncOp f) {
      for (NamedAttribute na : f->getAttrs()) {
        StringRef name = na.getName().getValue();
        if (matchTritonAttr(name)) {
          llvm::errs() << "Function illegal due to attribute: " << name << "\n";
          return false;
        }
      }
      return true;
    });

    addDynamicallyLegalOp<LLVM::AllocaOp>([](LLVM::AllocaOp allocaOp) {
      for (NamedAttribute na : allocaOp->getAttrs()) {
        StringRef name = na.getName().getValue();
        if (matchTritonAttr(name)) {
          llvm::errs() << "Alloca illegal due to attribute: " << name << "\n";
          return false; // Make it illegal, forcing conversion
        }
      }
      return true; // Legal if no triton attributes
    });
  }
};

struct ConvertTritonGPUToLLVM
    : public triton::impl::ConvertTritonGPUToLLVMBase<ConvertTritonGPUToLLVM> {
  using ConvertTritonGPUToLLVMBase::ConvertTritonGPUToLLVMBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::nvgpu::NVGPUDialect, LLVM::LLVMDialect,
                    NVVM::NVVMDialect>();
  }

  ConvertTritonGPUToLLVM(int32_t computeCapability)
      : ConvertTritonGPUToLLVMBase({computeCapability}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(mod);
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    // Lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      TritonGPUToLLVMTypeConverter typeConverter(context, option);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      mlir::triton::populateFuncOpConversionPattern(
          typeConverter, funcPatterns, numWarps, patternBenefitDefault);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(typeConverter);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    OpBuilder::InsertPoint indexInsertPoint;

    RewritePatternSet patterns(context);
    TargetInfo targetInfo(computeCapability);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMOptimizedPatterns(
        typeConverter, targetInfo, patterns,
        patternBenefitConvertLayoutOptimizedPattern);
    mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
    populateDotOpToLLVMPatterns(typeConverter, patterns, benefit);
    populateElementwiseOpToLLVMPatterns(typeConverter, patterns,
                                        axisInfoAnalysis, computeCapability,
                                        targetInfo, benefit);
    populateClampFOpToLLVMPattern(typeConverter, patterns, axisInfoAnalysis,
                                  computeCapability,
                                  patternBenefitClampOptimizedPattern);
    populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo, patterns,
                                      axisInfoAnalysis, benefit);
    mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
    populateBarrierOpToLLVMPatterns(typeConverter, patterns, benefit);
    populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns, benefit);
    populateClusterOpsToLLVMPatterns(typeConverter, patterns, benefit);
    mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                                    targetInfo, benefit);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     benefit);
    mlir::triton::NVIDIA::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                      benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
    mlir::triton::populateMemoryOpToLLVMPattern(typeConverter, targetInfo,
                                                patterns, benefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, benefit);
     

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

   
    // finalize to x86 

    // insert a few functions first for NVVM conversion
    insertStubFuncs();

    TritonGPULLVMToX86TypeConverter x86TypeConverter(context, option);
    RewritePatternSet patterns_strip(context);
    mlir::triton::populateStripGPUAndSetX86(x86TypeConverter, patterns_strip, targetInfo);
    
    X86LLVMConversionTarget x86ConvTarget(*context);
    if (failed(applyPartialConversion(mod, x86ConvTarget, std::move(patterns_strip))))
      return signalPassFailure();
    
    convertPtrAddressSpace(x86TypeConverter);

    // Fold CTAId when there is only 1 CTA.
    if (numCTAs == 1) {
      mod.walk([](triton::nvgpu::ClusterCTAIdOp id) {
        OpBuilder b(id);
        Value zero = LLVM::createConstantI32(id->getLoc(), b, 0);
        id.replaceAllUsesWith(zero);
      });
    }
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    auto global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::kSharedMemorySpace));
  }

  static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                              Type promotedType) {
    Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                  .cloneWith(std::nullopt, promotedType);
    return builder.create<triton::FpToFpOp>(loc, tensorPromotedType, operand);
  }

  void insertStubFuncs(){
        // In your pass’s runOnOperation():
    ModuleOp module = getOperation();
    auto *ctx = &getContext();
    Location loc = module.getLoc();

    // i32 () function type
    {
      auto i32Ty = IntegerType::get(ctx, 32);
      auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, /*params=*/{}, /*isVarArg=*/false);

      if (!module.lookupSymbol<LLVM::LLVMFuncOp>("nvgpu_cluster_id")) {
        OpBuilder builder(&getContext());
        builder.setInsertionPointToStart(module.getBody());
        auto stub = builder.create<LLVM::LLVMFuncOp>(loc, "nvgpu_cluster_id", fnTy);
        stub.setLinkage(LLVM::Linkage::External);
      }
    }

    {
      auto i32Ty = IntegerType::get(ctx, 32);
      auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, /*params=*/{i32Ty}, /*isVarArg=*/false);

      if (!module.lookupSymbol<LLVM::LLVMFuncOp>("nvvm_ctaid")) {
        OpBuilder builder(&getContext());
        builder.setInsertionPointToStart(module.getBody());
        auto stub = builder.create<LLVM::LLVMFuncOp>(loc, "nvvm_ctaid", fnTy);
        stub.setLinkage(LLVM::Linkage::External);
      }

      if (!module.lookupSymbol<LLVM::LLVMFuncOp>("nvvm_tid")) {
        OpBuilder builder(&getContext());
        builder.setInsertionPointToStart(module.getBody());
        auto stub = builder.create<LLVM::LLVMFuncOp>(loc, "nvvm_tid", fnTy);
        stub.setLinkage(LLVM::Linkage::External);
      }

    }

    {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto fnTy = LLVM::LLVMFunctionType::get(voidTy, /*params=*/{}, /*isVarArg=*/false);
      if (!module.lookupSymbol<LLVM::LLVMFuncOp>("nvvm_barrier0")) {
        OpBuilder builder(&getContext());
        builder.setInsertionPointToStart(module.getBody());
        auto stub = builder.create<LLVM::LLVMFuncOp>(loc, "nvvm_barrier0", fnTy);
        stub.setLinkage(LLVM::Linkage::External);
      }
    }
    {
      auto i64Ty = IntegerType::get(ctx, 64);
      auto ptrTy = LLVM::LLVMPointerType::get(ctx, /*addrSpace=*/0);
      auto fnTy = LLVM::LLVMFunctionType::get(ptrTy, /*params=*/{i64Ty, ptrTy}, /*isVarArg=*/false);
      
      if (!module.lookupSymbol<LLVM::LLVMFuncOp>("metrics_alloca")) {
        OpBuilder builder(&getContext());
        builder.setInsertionPointToStart(module.getBody());
        auto stub = builder.create<LLVM::LLVMFuncOp>(loc, "metrics_alloca", fnTy);
        stub.setLinkage(LLVM::Linkage::External);
      }
    }
    {
      auto i64Ty = IntegerType::get(ctx, 64);
      auto fnTy = LLVM::LLVMFunctionType::get(i64Ty, /*params=*/{}, /*isVarArg=*/false);
      
      if (!module.lookupSymbol<LLVM::LLVMFuncOp>("metrics_dummy")) {
        OpBuilder builder(&getContext());
        builder.setInsertionPointToStart(module.getBody());
        auto stub = builder.create<LLVM::LLVMFuncOp>(loc, "metrics_dummy", fnTy);
        stub.setLinkage(LLVM::Linkage::External);
      }
    }
  }

  void convertPtrAddressSpace(LLVMTypeConverter &typeConverter) {
    ModuleOp module = getOperation();
    auto &ctx = getContext();

    module.walk([&](Operation *op) {
      // 1) Normalize all result types through the converter
      for (OpResult result : op->getResults()) {
        Type oldTy = result.getType();
        if (auto newTy = typeConverter.convertType(oldTy)) {
          if (newTy != oldTy){
            // llvm::errs() << "Converting type: " << oldTy << " to " << newTy
            //           << "\n";
            result.setType(newTy);
          }
        }
      }

      // 2) Normalize all operand types through the converter
      for (OpOperand &operand : op->getOpOperands()) {
        Type oldTy = operand.get().getType();
        if (auto newTy = typeConverter.convertType(oldTy)) {
          if (newTy != oldTy){
              // llvm::errs() << "Converting type: " << oldTy << " to " << newTy
              //         << "\n";
            operand.get().setType(newTy);

          }
        }
      }

      if (auto glob = dyn_cast<LLVM::GlobalOp>(op)) {
        if(glob.getAddrSpace() != 0){
          llvm::errs() << "Converting global: " << glob.getName() << "\n";
          glob.setAddrSpace(0);
        }
      }

    });
  }

  void insertBranchAssertions(){

  }

// mjc
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass() {
  return std::make_unique<ConvertTritonGPUToLLVM>();
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability);
}

} // namespace triton
} // namespace mlir
