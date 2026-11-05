
#include "ConvertDotCommon.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/X86Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace mlir::triton::cpu {
#define GEN_PASS_DEF_CONVERTDOTTOX86
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace mlir::triton::cpu

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

namespace {
struct ConvertDotToContract : public OpRewritePattern<cpu::DotOp> {
  using OpRewritePattern<cpu::DotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cpu::DotOp op,
                                PatternRewriter &rewriter) const override {
    Value a = op.getA();
    Value b = op.getB();
    Value c = op.getC();
    VectorType aType = dyn_cast<VectorType>(a.getType());
    VectorType bType = dyn_cast<VectorType>(b.getType());
    VectorType cType = dyn_cast<VectorType>(c.getType());

    if (!aType || !bType || !cType || aType.getRank() != 2 ||
        bType.getRank() != 2 || cType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected 2D vectors");

    MLIRContext *ctx = rewriter.getContext();
    ArrayAttr indexingMaps, iteratorTypes;
    auto aMap = AffineMap::getMultiDimMapWithTargets(3, {0, 2}, ctx);
    auto bMap = AffineMap::getMultiDimMapWithTargets(3, {2, 1}, ctx);
    auto cMap = AffineMap::getMultiDimMapWithTargets(3, {0, 1}, ctx);
    indexingMaps = rewriter.getAffineMapArrayAttr({aMap, bMap, cMap});
    iteratorTypes = rewriter.getArrayAttr(
        {vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction)});
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        op, a, b, c, indexingMaps, iteratorTypes);

    return success();
  }
};

struct PromoteLoopIVToIndex : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    Value iv = op.getInductionVar();
    if (iv.getType().isIndex())
      return rewriter.notifyMatchFailure(
          op, "induction variable is already index-typed");

    if (DataLayout::closest(op).getTypeSizeInBits(rewriter.getIndexType()) <
        iv.getType().getIntOrFloatBitWidth())
      return rewriter.notifyMatchFailure(
          op, "integer type is wider than index type");

    for (Operation *user : iv.getUsers()) {
      if (!isa<arith::IndexCastOp>(user))
        return rewriter.notifyMatchFailure(
            op, "induction variable has non-index cast users");
    }

    // Check some necessary conditions for VCMLU and nanokernels
    if (op.getNumRegionIterArgs() != 1 &&
        range_size(op.getBody()->getOps<vector::ContractionOp>()) != 1)
      return rewriter.notifyMatchFailure(
          op, "expected exactly one loop-carried value and one contraction op");

    MLIRContext *ctx = rewriter.getContext();
    Location loc = op.getLoc();

    rewriter.setInsertionPoint(op);
    Value lb = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), op.getLowerBound());
    Value ub = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), op.getUpperBound());
    Value step = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), op.getStep());

    rewriter.modifyOpInPlace(op, [&] {
      op.setLowerBound(lb);
      op.setUpperBound(ub);
      op.setStep(step);
      cast<BlockArgument>(iv).setType(rewriter.getIndexType());
    });

    rewriter.setInsertionPointToStart(op.getBody());
    Value ivInt = arith::IndexCastOp::create(rewriter, loc, iv.getType(), iv);
    rewriter.replaceAllUsesExcept(iv, ivInt, ivInt.getDefiningOp());
    for (Operation *user : iv.getUsers())
      if (auto cast = dyn_cast<arith::IndexCastOp>(user))
        rewriter.replaceOp(cast, iv);

    return success();
  }
};

struct FoldShapeCastIntoTransferWrite
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getPermutationMap().isMinorIdentity() || op.getMask() ||
        op.hasOutOfBoundsDim())
      return failure();

    auto castOp = op.getValueToStore().getDefiningOp<vector::ShapeCastOp>();
    if (!castOp || !castOp->hasOneUse())
      return failure();

    VectorType srcTy = castOp.getSourceVectorType();
    VectorType resTy = castOp.getResultVectorType();

    if (srcTy.getRank() >= resTy.getRank())
      return failure();

    // Check all added dimensions have size 1.
    if (!all_of(drop_end(resTy.getShape(), srcTy.getRank()),
                [](int64_t val) { return val == 1; }))
      return failure();

    // Check shape suffix matches the source type.
    if (!equal(drop_begin(resTy.getShape(), resTy.getRank() - srcTy.getRank()),
               srcTy.getShape()))
      return failure();

    SmallVector<bool> inBounds(srcTy.getRank(), true);
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        op, castOp.getSource(), op.getBase(), op.getIndices(), inBounds);

    return success();
  }
};

template <typename OpTy>
struct MoveAfterProducer : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!isMemoryEffectFree(op) ||
        !op->template hasTrait<OpTrait::AlwaysSpeculatableImplTrait>() ||
        op->getNumOperands() != 1)
      return failure();

    Operation *operandDef = op->getOperand(0).getDefiningOp();
    if (!operandDef || !operandDef->hasOneUse() ||
        op->getPrevNode() == operandDef)
      return failure();

    rewriter.moveOpAfter(op, operandDef);

    return success();
  }
};

struct HoistAlloca : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp op,
                                PatternRewriter &rewriter) const override {
    auto func = op->getParentOfType<triton::FuncOp>();
    if (!func)
      return failure();
    if (op->getParentOp() == func || !op.getDynamicSizes().empty())
      return failure();

    Block *entry = &func.getBody().front();
    rewriter.moveOpBefore(op, entry, entry->begin());
    rewriter.modifyOpInPlace(op, [&]() { op.setAlignment(64); });

    return success();
  }
};

class AllocaCollector : public RewriterBase::Listener {
public:
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override {
    // Ignore pure moves of ops that already existed.
    if (previous.isSet())
      return;
    if (isa<memref::AllocaOp>(op))
      allocas.insert(op);
  }

  void notifyOperationErased(Operation *op) override {
    if (isa<memref::AllocaOp>(op))
      allocas.remove(op);
  }

  void clear() { allocas.clear(); }

  SmallVector<Operation *> getAllocas() { return allocas.takeVector(); }

private:
  SetVector<Operation *> allocas;
};

struct ConvertDotToX86
    : public triton::cpu::impl::ConvertDotToX86Base<ConvertDotToX86> {
  ConvertDotToX86(std::string cpuFeatures) { this->cpuFeatures = cpuFeatures; }

  void runOnOperation() override {
    LDBG("[DotToX86] Attempting application of x86 dialect patterns");

    ModuleOp module = getOperation();
    if (cpuFeatures.find("amx-bf16") == std::string::npos &&
        cpuFeatures.find("amx-int8") == std::string::npos) {
      LDBG("[DotToX86] Target has no AMX, skipping");
      return;
    }

    SetVector<triton::FuncOp> candidates;
    module.walk([&](triton::cpu::DotOp dot) {
      auto parent = dot->getParentOfType<triton::FuncOp>();
      assert(parent && "triton::cpu::DotOp must be inside a triton::FuncOp");
      candidates.insert(parent);
    });
    if (candidates.empty()) {
      LDBG("[DotToX86] No triton_dot.cpu ops found, skipping");
      return;
    }

    MLIRContext *ctx = &getContext();
    RewritePatternSet prepPatterns(ctx);
    prepPatterns.add<ConvertDotToContract>(ctx);
    prepPatterns.add<PromoteLoopIVToIndex>(ctx);
    prepPatterns.add<FoldShapeCastIntoTransferWrite>(ctx);
    prepPatterns.add<MoveAfterProducer<cpu::ExtractMemRefOp>>(ctx);
    prepPatterns.add<MoveAfterProducer<arith::IndexCastOp>>(ctx);

    RewritePatternSet patterns(ctx);
    x86::populatePrepareVectorContractLoopsPatterns(patterns);
    x86::populateVectorContractMultiLevelUnrollForAMXPatterns(patterns);
    x86::populateVectorContractToAMXDotProductPatterns(patterns);

    RewritePatternSet postPatterns(ctx);
    postPatterns.add<HoistAlloca>(ctx);

    FrozenRewritePatternSet frozenPrepPatterns(std::move(prepPatterns));
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    FrozenRewritePatternSet frozenPostPatterns(std::move(postPatterns));
    GreedyRewriteConfig withCSE;
    withCSE.enableCSEBetweenIterations();

    AllocaCollector allocaCollector;
    GreedyRewriteConfig withAllocaListener;
    withAllocaListener.setListener(&allocaCollector);

    for (auto func : candidates) {
      LDBG("[DotToX86] IR before prep patterns:\n" << func << '\n');

      func.walk([&](scf::ForOp loop) { moveLoopInvariantCode(loop); });

      if (failed(applyPatternsGreedily(func, frozenPrepPatterns, withCSE)))
        signalPassFailure();

      LDBG("[DotToX86] IR after prep patterns:\n" << func << '\n');

      allocaCollector.clear();
      if (failed(
              applyPatternsGreedily(func, frozenPatterns, withAllocaListener)))
        signalPassFailure();

      LDBG("[DotToX86] IR after main patterns:\n" << func << '\n');

      SmallVector<Operation *> allocas = allocaCollector.getAllocas();
      std::reverse(allocas.begin(), allocas.end());
      if (failed(applyOpPatternsGreedily(allocas, frozenPostPatterns)))
        signalPassFailure();

      LDBG("[DotToX86] IR after post patterns:\n" << func << '\n');
    }
  }
};

} // namespace

namespace mlir::triton::cpu {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotToX86(std::string cpuFeatures) {
  return std::make_unique<ConvertDotToX86>(cpuFeatures);
}

} // namespace mlir::triton::cpu
