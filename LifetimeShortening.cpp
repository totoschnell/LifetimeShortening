#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Analysis/SliceWalk.h"
#include "mlir/Transforms/RegionUtils.h"

//#define DEBUG_PROVENANCE
#ifdef DEBUG_PROVENANCE
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "cudaq/Support/Plugin.h"
#endif

#ifdef DEBUG_LIFETIME

#endif

#define RELEASE
#ifdef RELEASE
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "cudaq/Support/Plugin.h"
#endif

#define DEBUG_TYPE "lifetime-shortening"

using namespace mlir;

namespace {
using Node = std::pair<Value, unsigned>;
constexpr unsigned WHOLE = std::numeric_limits<unsigned>::max();
using LivingRange = std::optional<std::pair<Operation *, Operation *>>;

bool isQuantumReferenceType(Type ty){
    return isa<cudaq::quake::RefType, cudaq::quake::VeqType, cudaq::quake::StruqType>(ty);
}

bool isCallableType(Type ty){
    return isa<cudaq::cc::CallableType>(ty);
}

//general quantum resource alias analysis for memory-semantic quake IR
class Provenance {
private:
    //the refs in the set are derived from the key
    DenseMap<Node, DenseSet<Node>> deriveGraph;
    //root references to quantum resources which are allocated inside this kernel
    DenseSet<Value> internalProvenanceRoots;
    //root references to quantum resources which are passed to this kernel as arguments
    DenseSet<Value> externalProvenanceRoots;
    //root references to quantum resources which are passed to lambda as arguments
    DenseSet<Value> lambdaExternalProvenanceRoots;

    void reportError(Operation *op, const llvm::Twine &errorMsg, Location errorLocation){
        existQValueWithoutReasonableProvenance = true;
        op->emitError(errorMsg).attachNote(errorLocation);
        deriveGraph.clear();
        internalProvenanceRoots.clear();
        externalProvenanceRoots.clear();
        lambdaExternalProvenanceRoots.clear();
    }

    //struq reference definition coming from the control flow, from block arguments or from results of cc.if or cc.loop, track the provenance of the members
    void handleStruqFromControlFlow(Value &derivedStruq, Value &sourceStruq){
        auto struqType = dyn_cast<cudaq::quake::StruqType>(derivedStruq.getType());
        for(unsigned index = 0; index < struqType.getNumMembers(); ++index){
            deriveGraph[Node(sourceStruq, index)].insert(Node(derivedStruq, index));
        }
    }

    //process quantum reference from op results
    void processOpQRefResult(Value &v, Operation *op){
        if(auto alloca = dyn_cast<cudaq::quake::AllocaOp>(op)){
            if(!alloca.hasInitializedState()){
                internalProvenanceRoots.insert(v);
                //struq cannot be initialized according to QuakeOp.td
                if(auto struqTp = dyn_cast<cudaq::quake::StruqType>(v.getType())){
                    size_t numOfMembers = struqTp.getNumMembers();
                    for(size_t i = 0; i < numOfMembers; ++i){
                        deriveGraph[Node(v, WHOLE)].insert(Node(v, i));
                    }
                }
            }
        }
        else if(isa<cudaq::quake::InitializeStateOp>(op)){
            internalProvenanceRoots.insert(v);
        }
        else if(auto concat = dyn_cast<cudaq::quake::ConcatOp>(op)){
            for(Value component : concat.getOperands()){
                deriveGraph[Node(component, WHOLE)].insert(Node(v, WHOLE));
            }
        }
        else if(auto extractRef = dyn_cast<cudaq::quake::ExtractRefOp>(op)){
            deriveGraph[Node(extractRef.getOperand(0), WHOLE)].insert(Node(v, WHOLE));
        }
        else if(auto relaxSize = dyn_cast<cudaq::quake::RelaxSizeOp>(op)){
            deriveGraph[Node(relaxSize.getOperand(), WHOLE)].insert(Node(v, WHOLE));
        }
        else if(auto subveq = dyn_cast<cudaq::quake::SubVeqOp>(op)){
            deriveGraph[Node(subveq.getOperand(0), WHOLE)].insert(Node(v, WHOLE));
        }
        else if(auto make_struq = dyn_cast<cudaq::quake::MakeStruqOp>(op)){
            unsigned memberIdx = 0;
            for(Value member : make_struq.getOperands()){
                //the result struq is alias to all of the operands, because if there is a use of the result struq(eg. as kernel call param), then the resources building this struq have to be kept alive
                deriveGraph[Node(member, WHOLE)].insert(Node(v, WHOLE));
                deriveGraph[Node(member, WHOLE)].insert(Node(v, memberIdx++));
            }
        }
        //compiler frontend will reject embedded struq definition, so we can suppose the member of the struq is quake.ref or quake.veq, and the resulting type should match
        else if(auto getMember = dyn_cast<cudaq::quake::GetMemberOp>(op)){
            Value struq = getMember.getStruq();
            uint32_t index = getMember.getIndex();
            //here we only alias member to member, because a use of the result does not necessarily need all members of the source struq to be alive
            deriveGraph[Node(struq, index)].insert(Node(v, WHOLE));
        }
        //this quantum reference is a result of cc.loop op or cc.if op
        else if(isa<cudaq::cc::IfOp, cudaq::cc::LoopOp>(op)){
            std::optional<llvm::SmallVector<Value>> predecessors =  getControlFlowPredecessors(v);
            if(!predecessors.has_value()){
                reportError(op, "Detected quantum reference as op result without proper predecessors", v.getLoc());
            }
            else{
                for(Value predecessor : predecessors.value()){
                    deriveGraph[Node(predecessor, WHOLE)].insert(Node(v, WHOLE));
                    if(isa<cudaq::quake::StruqType>(v.getType())){
                        handleStruqFromControlFlow(v, predecessor);
                    }
                }
            }
        }
        else{
            //unknown operation returned a quantum reference, emit an error
            reportError(op, "Unknown op has returned a quantum reference", v.getLoc());
        }
    }

    //process closure from op results
    void processOpClosureResult(Value v, Operation *op){
        //known closure is created
        if(auto createLamba = dyn_cast<cudaq::cc::CreateLambdaOp>(op)){
            Region &lambdaRegion = createLamba.getRegion();
            SetVector<Value> capturedValues;
            getUsedValuesDefinedAbove(lambdaRegion, lambdaRegion, capturedValues);
            for(const Value &capturedValue : capturedValues){
                if(isQuantumReferenceType(capturedValue.getType()) || isCallableType(capturedValue.getType())){
                    deriveGraph[Node(capturedValue, WHOLE)].insert(Node(v, WHOLE));
                }
            }
        }
        else if(isa<cudaq::cc::IfOp, cudaq::cc::LoopOp>(op)){
            std::optional<llvm::SmallVector<Value>> predecessors =  getControlFlowPredecessors(v);
            if(!predecessors.has_value()){
                reportError(op, "Detected callable as op result without proper predecessors", v.getLoc());
            }
            else{
                for(Value predecessor : predecessors.value()){
                    deriveGraph[Node(predecessor, WHOLE)].insert(Node(v, WHOLE));
                }
            }
        }
        //unknown operation returned a closure, emit an error
        else{
            reportError(op, "Unknown op has returned a callable", v.getLoc());
        }
    }

    //process quantum reference and callable from block arguments
    void processBlockArgument(BlockArgument blockArgument){
        std::optional<llvm::SmallVector<Value>> predecessors = getControlFlowPredecessors(blockArgument);
        if(!predecessors.has_value()){
            Block *block = blockArgument.getOwner();
            //kernel function argument
            if(block->isEntryBlock() && isa<func::FuncOp>(block->getParentOp())){
                externalProvenanceRoots.insert(blockArgument);
                if(auto struqType = dyn_cast<cudaq::quake::StruqType>(blockArgument.getType())){
                    for(unsigned index = 0; index < struqType.getNumMembers(); ++index){
                        deriveGraph[Node(blockArgument, WHOLE)].insert(Node(blockArgument, index));
                    }
                }
            }
            //lambda arguments
            else if(block->isEntryBlock() && isa<cudaq::cc::CreateLambdaOp>(block->getParentOp())){
                lambdaExternalProvenanceRoots.insert(blockArgument);
                if(auto struqType = dyn_cast<cudaq::quake::StruqType>(blockArgument.getType())){
                    for(unsigned index = 0; index < struqType.getNumMembers(); ++index){
                        deriveGraph[Node(blockArgument, WHOLE)].insert(Node(blockArgument, index));
                    }
                }
            }
            else{
                reportError(block->getParentOp(), "Detected quantum reference or callable as block argument without reasonable provenance. Possible dead block in IR detected.", blockArgument.getLoc());
            }
        }
        else{
            for(Value predecessor : predecessors.value()){
                deriveGraph[Node(predecessor, WHOLE)].insert(Node(blockArgument, WHOLE));
                if(isa<cudaq::quake::StruqType>(blockArgument.getType())){
                    handleStruqFromControlFlow(blockArgument, predecessor);
                }
            }
        }
    }

    //walk the func.func op tree
    void buildDeriveGraph(func::FuncOp func){
        func.walk<WalkOrder::PostOrder>([&](Operation *op)->WalkResult{
            for(Value v : op->getResults()){
                if(isQuantumReferenceType(v.getType())){
                    processOpQRefResult(v, op);
                }
                else if(isCallableType(v.getType())){
                    processOpClosureResult(v, op);
                }
                if(existQValueWithoutReasonableProvenance){
                    return WalkResult::interrupt();
                }
            }
            for(Region &r : op->getRegions()){
                for(Block &b : r.getBlocks()){
                    for(BlockArgument blockArgument : b.getArguments()){
                        if(isQuantumReferenceType(blockArgument.getType()) || isCallableType(blockArgument.getType())){
                            processBlockArgument(blockArgument);
                        }
                        if(existQValueWithoutReasonableProvenance){
                            return WalkResult::interrupt();
                        }
                    }
                }
            }
            return WalkResult::advance();
        });
    }

    void buildProvenanceResult(DenseSet<Value> &rootSet, DenseMap<Value, DenseSet<Value>> &analysisResult){
        if(existQValueWithoutReasonableProvenance){
            return;
        }
        for(Value root : rootSet){
            analysisResult[root];
            SetVector<Node> workingSet;
            //the root is an alias of itself
            workingSet.insert(Node(root, WHOLE));
            size_t head = 0;
            while(head != workingSet.size()){
                Node current = workingSet[head++];
                if(current.second == WHOLE){
                    analysisResult[root].insert(current.first);
                }
                DenseMap<Node, DenseSet<Node>>::const_iterator it = deriveGraph.find(current);
                if(it == deriveGraph.end()){
                    continue;
                }
                for(const Node &derivedNode : it->second){
                    workingSet.insert(derivedNode);
                }
            }
        }
    }

public:
    bool existQValueWithoutReasonableProvenance;
    DenseMap<Value, DenseSet<Value>> internalRootProvenanceAnalysisResult;
    DenseMap<Value, DenseSet<Value>> externalRootProvenanceAnalysisResult;
    DenseMap<Value, DenseSet<Value>> lambdaExternalRootProvenanceAnalysisResult;
   
    Provenance(func::FuncOp func){
        existQValueWithoutReasonableProvenance = false;
        buildDeriveGraph(func);
        buildProvenanceResult(internalProvenanceRoots, internalRootProvenanceAnalysisResult);
        //could be turned off for intra-procedural lifetime shortening, PointsTo analysis only
        buildProvenanceResult(externalProvenanceRoots, externalRootProvenanceAnalysisResult);
        //could be turned off for intra-procedural lifetime shortening, PointsTo analysis only
        buildProvenanceResult(lambdaExternalProvenanceRoots, lambdaExternalRootProvenanceAnalysisResult);
    }
};

class PointsTo{
    private:
    //build "points-to analysis result", analyse which quantum reference points to which resource, callables is not a quantum reference type, it is a classical type, it is thus excluded
    void buildPointsToResult(Provenance &provenanceResult){
        if(provenanceResult.existQValueWithoutReasonableProvenance){
            this->existQValueWithoutReasonableProvenance = true;
            return;
        }
        DenseMap<Value, DenseSet<Value>> mergedResult;
        mergedResult.insert(provenanceResult.internalRootProvenanceAnalysisResult.begin(), provenanceResult.internalRootProvenanceAnalysisResult.end());
        mergedResult.insert(provenanceResult.externalRootProvenanceAnalysisResult.begin(), provenanceResult.externalRootProvenanceAnalysisResult.end());
        //currently, lambda interprocedural alias analysis is NOT performed, because the parameters of lambda will be bound to different value at different calls. interprocedural alias analysis could be more precise if it is applied to specific lambda use.
        //anyway, alias inside a lambda region could be analyzed
        mergedResult.insert(provenanceResult.lambdaExternalRootProvenanceAnalysisResult.begin(), provenanceResult.lambdaExternalRootProvenanceAnalysisResult.end());
        for(DenseMap<Value, DenseSet<Value>>::const_iterator it = mergedResult.begin(); it != mergedResult.end(); ++it){
            for(const Value &refOrCallable : it->getSecond()){
                if(isQuantumReferenceType(refOrCallable.getType())){
                    pointsToResult[refOrCallable].insert(it->getFirst());
                }
            }
        }
    }

    public:
    bool existQValueWithoutReasonableProvenance = false;
    DenseMap<Value, DenseSet<Value>> pointsToResult;
    PointsTo(Provenance &provenanceResult){
        buildPointsToResult(provenanceResult);
    }
};

//this is a intraprocedural AA for quantum references
class AA{
    private:
    DenseMap<Value, DenseSet<Value>> pointsToResult;
    DenseMap<Value, DenseSet<Value>> mayAliasCache;

    public:
    AA(PointsTo &pointsTo){
        pointsToResult = pointsTo.pointsToResult;
    }
    //alias query
    std::optional<bool> mayAlias(Value v1, Value v2){
        if(!isQuantumReferenceType(v1.getType()) && !isQuantumReferenceType(v2.getType())){
            //unknown value type, return nullopt
            return std::nullopt;
        }
        else if(isQuantumReferenceType(v1.getType()) != isQuantumReferenceType(v2.getType())){
            //testing a quantum reference type with another type, they must not be alias
            return std::optional(false);
        }
        else{
            //TODO: test if there is intersection between the root sets of v1 and v2
        }
        return std::nullopt;
    }
};

class Lifetime{
private:
    DenseMap<Value, LivingRange> aliasLivingRangeCache;
    LivingRange findOrCreateAliasLivingRangeCache(Value alias){
        LivingRange aliasLivingRange;
        DenseMap<Value, LivingRange>::const_iterator existingAliasLivingRangeCache = aliasLivingRangeCache.find(alias);
        //if no living range info exists in cache, then create this info. Otherwise use the cached info instead
        if(existingAliasLivingRangeCache == aliasLivingRangeCache.end()){
            if(alias.use_empty()){
                aliasLivingRange = std::nullopt;
            }
            else{
                Block *aliasParentBlock = alias.getParentBlock();
                Operation *firstUseOp = nullptr;
                Operation *lastUseOp = nullptr;
                for(Operation *user : alias.getUsers()){
                    Operation *ancestor = aliasParentBlock->findAncestorOpInBlock(*user);
                    if(!ancestor){
                        //found a use out of define scope
                        malformedIR = true;
                        user->emitError("Detected quantum reference use out of its defining scope").attachNote(alias.getLoc());
                        aliasLivingRangeCache.clear();
                        internalRootLivingRange.clear();
                        return std::nullopt;
                    }
                    if(!firstUseOp || ancestor->isBeforeInBlock(firstUseOp)){
                        firstUseOp = ancestor;
                    }
                    if(!lastUseOp || lastUseOp->isBeforeInBlock(ancestor)){
                        lastUseOp = ancestor;
                    }
                }
                aliasLivingRange = std::make_pair(firstUseOp, lastUseOp);
            }
            aliasLivingRangeCache[alias] = aliasLivingRange;
        }
        else{
            aliasLivingRange = existingAliasLivingRangeCache->second;
        }
        return aliasLivingRange;
    }

public:
    bool malformedIR;
    bool detectedDanglingReference;
    DenseMap<Value, LivingRange> internalRootLivingRange;
    Lifetime(DenseMap<Value, DenseSet<Value>> &internalRootProvenanceAnalysisResult){
        malformedIR = false;
        detectedDanglingReference = false;
        for(DenseMap<Value, DenseSet<Value>>::const_iterator it = internalRootProvenanceAnalysisResult.begin(); it != internalRootProvenanceAnalysisResult.end(); ++it){
            Value currentRoot = it->first;
            //No living range for this root at the very beginning
            Operation *firstDerivedReferenceUse = nullptr;
            Operation *lastDerivedReferenceUse = nullptr;
            Block *currentRootBlock = currentRoot.getParentBlock();
            LivingRange currentRootLivingRange;
            for(Value alias : it->second){
                //extend living range for this root according to living range of its aliases
                LivingRange aliasLivingRange = findOrCreateAliasLivingRangeCache(alias);
                if(malformedIR){
                    return;
                }
                if(aliasLivingRange.has_value()){
                    Operation *firstUseInRootBlock = currentRootBlock->findAncestorOpInBlock(*(aliasLivingRange.value().first));
                    Operation *lastUseInRootBlock = currentRootBlock->findAncestorOpInBlock(*(aliasLivingRange.value().second));
                    if(!firstUseInRootBlock || !lastUseInRootBlock){
                        //here a possible dangling quantum reference is found. This is an undefined behavior. Report this in console. 
                        detectedDanglingReference = true;
                        Operation *escapingUse = !firstUseInRootBlock ? aliasLivingRange.value().first : aliasLivingRange.value().second;
                        escapingUse->emitError("detected a possible dangling reference use").attachNote(currentRoot.getLoc()) << "quantum resource originates here";
                        internalRootLivingRange.clear();
                        aliasLivingRangeCache.clear();
                        return;
                    }
                    if(!firstDerivedReferenceUse || firstUseInRootBlock->isBeforeInBlock(firstDerivedReferenceUse)){
                        firstDerivedReferenceUse = firstUseInRootBlock;
                    }
                    if(!lastDerivedReferenceUse || lastDerivedReferenceUse->isBeforeInBlock(lastUseInRootBlock)){
                        lastDerivedReferenceUse = lastUseInRootBlock;
                    }
                }
            }
            if(!firstDerivedReferenceUse || !lastDerivedReferenceUse){
                //literally no use for this root
                currentRootLivingRange = std::nullopt;
            }
            else{
                currentRootLivingRange = std::make_pair(firstDerivedReferenceUse, lastDerivedReferenceUse);
            }
            internalRootLivingRange[currentRoot] = currentRootLivingRange;
        }
    }
};

#ifdef DEBUG_PROVENANCE
//generated by ChatGPT
// Standalone analysis printer. No lifetime shortening is performed here.
static void buildValueOrder(func::FuncOp func,
                           SetVector<Value> &valueOrder) {
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        for (Value value : op->getResults()) {
            valueOrder.insert(value);
        }
        for (Region &region : op->getRegions()) {
            for (Block &block : region.getBlocks()) {
                for (BlockArgument argument : block.getArguments()) {
                    valueOrder.insert(argument);
                }
            }
        }
    });
}

static void printProvenanceValue(Value value, AsmState &asmState,
                                 llvm::raw_ostream &os) {
    value.printAsOperand(os, asmState);
    os << " : ";
    value.getType().print(os);
    if (isCallableType(value.getType())) {
        os << " [callable dependency]";
    } else if (isQuantumReferenceType(value.getType())) {
        os << " [quantum reference]";
    }
}

static void printProvenanceRoots(
    StringRef label, const DenseMap<Value, DenseSet<Value>> &result,
    const SetVector<Value> &valueOrder, AsmState &asmState,
    llvm::raw_ostream &os) {
    os << "\n--- " << label << " (" << result.size() << ") ---\n";
    if (result.empty()) {
        os << "  (none)\n";
        return;
    }

    for (size_t rootIndex = 0; rootIndex < valueOrder.size(); ++rootIndex) {
        Value root = valueOrder[rootIndex];
        DenseMap<Value, DenseSet<Value>>::const_iterator entry =
            result.find(root);
        if (entry == result.end()) {
            continue;
        }

        os << "\nRoot: [v" << rootIndex << "] ";
        printProvenanceValue(root, asmState, os);
        os << "\nProvenance: {\n";

        for (size_t derivedIndex = 0; derivedIndex < valueOrder.size();
             ++derivedIndex) {
            Value derived = valueOrder[derivedIndex];
            if (!entry->second.count(derived)) {
                continue;
            }
            os << "  [v" << derivedIndex << "] ";
            printProvenanceValue(derived, asmState, os);
            if (derived == root) {
                os << " [root]";
            }
            os << "\n";
        }
        os << "}\n";
    }
}

struct TestProvenancePass
    : public PassWrapper<TestProvenancePass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestProvenancePass)

    StringRef getArgument() const override {
        return "test-quake-provenance";
    }

    StringRef getDescription() const override {
        return "Print quantum-reference and callable provenance without changing IR";
    }

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        if (func.isExternal()) {
            markAllAnalysesPreserved();
            return;
        }

        llvm::raw_ostream &os = llvm::outs();
        os << "\n=== Provenance Analysis: " << func.getSymName() << " ===\n";
        Provenance analysis(func);
        if (analysis.existQValueWithoutReasonableProvenance) {
            os << "ERROR: Unknown quantum-reference or callable provenance\n";
            os.flush();
            signalPassFailure();
            return;
        }

        AsmState asmState(func.getOperation());
        SetVector<Value> valueOrder;
        buildValueOrder(func, valueOrder);

        printProvenanceRoots("Internal allocation roots",
                             analysis.internalRootProvenanceAnalysisResult,
                             valueOrder, asmState, os);
        printProvenanceRoots("Function argument roots",
                             analysis.externalRootProvenanceAnalysisResult,
                             valueOrder, asmState, os);
        printProvenanceRoots("Lambda argument roots",
                             analysis.lambdaExternalRootProvenanceAnalysisResult,
                             valueOrder, asmState, os);

        os << "\n=== End Provenance Analysis ===\n";
        os.flush();
        markAllAnalysesPreserved();
    }
};
//end of code generated by ChatGPT
#endif

#ifdef RELEASE
struct LifetimeShortening : public PassWrapper<LifetimeShortening, OperationPass<func::FuncOp>>{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LifetimeShortening)

    StringRef getArgument() const override{
        return "lifetime-shortening";
    }

    StringRef getDescription() const override{
        return "Do provenance analysis and lifetime analysis. Then adjust the location of quake.alloca and quake.dealloc accordingly";
    }

    //the correctness of lifetime shortening analysis relies on a precondition: The IR must only consist of structured ops. This implies that there is only one block 
    //in every region, because cc dialect has no op to br from one block to another. There should be no existing deallocs in the IR. If unwind-by-dataflow 
    //is used to replace unwind-lowering, and this pass is placed in front of add-deallocs, the precondition should hold for IRs generated by compiler front-end.
    bool checkSingleBlockRegionPrecondition(){
        func::FuncOp func = getOperation();
        bool checkResult = true;
        func.walk<WalkOrder::PostOrder>([&](Operation *op)->WalkResult{
            if(isa<cudaq::quake::DeallocOp>(op)){
                checkResult = false;
                return WalkResult::interrupt();
            }
            for(Region &r : op->getRegions()){
                if(r.getBlocks().size() > 1){
                    checkResult = false;
                    return WalkResult::interrupt();
                }
            }
            return WalkResult::advance();
        });
        return checkResult;
    }
    
    bool checkLifetimeShorteningPrecondition(){
        //maybe there will be some other preconditions in the future
        return checkSingleBlockRegionPrecondition();
    }

    //currently we are not keeping the allocation order of quantum resources according to the definition order in the source IR
    void runOnOperation() override{
        if(!checkLifetimeShorteningPrecondition()){
            //if precondition check failed, skip this pass, use default add-deallocs pass instead
            return;
        }
        func::FuncOp func = getOperation();
        Provenance provenanceResult(func);
        if(provenanceResult.existQValueWithoutReasonableProvenance){
            signalPassFailure();
            return;
        }
        Lifetime lifetimeResult(provenanceResult.internalRootProvenanceAnalysisResult);
        if(lifetimeResult.malformedIR || lifetimeResult.detectedDanglingReference){
            signalPassFailure();
            return;
        }
        for(DenseMap<Value, LivingRange>::const_iterator it = lifetimeResult.internalRootLivingRange.begin(); it != lifetimeResult.internalRootLivingRange.end(); ++it){
            Value currentRoot = it->first;
            LivingRange currentRootLivingRange = it->second;
            Operation *definingOp = currentRoot.getDefiningOp();
            if(!currentRootLivingRange.has_value()){
                //no use for this root, erase resource allocation for this root
                if(auto initStateOp = dyn_cast<cudaq::quake::InitializeStateOp>(definingOp)){
                    //if this root is a return value of quake.initializeState，then erase this initialize state and the resource allocation accordingly
                    currentRoot = initStateOp.getOperand(0);
                    definingOp->erase();
                    definingOp = currentRoot.getDefiningOp();
                }
                definingOp->erase();
            }
            else{
                Operation *firstUse = currentRootLivingRange.value().first;
                Operation *lastUse = currentRootLivingRange.value().second;
                if(!isa<cudaq::quake::InitializeStateOp>(definingOp)){
                    //we are not moving initializeState op before the first use of a quantum reference because in IR a state is not an SSA value, it's a memory value.
                    //if we want to move this initializeState op, then we have to prove this state has not been overwritten.
                    //thus we ignore moving this op for now
                    definingOp->moveBefore(firstUse);
                }
                OpBuilder builder(lastUse->getContext());
                if(isa<cudaq::cc::ContinueOp>(lastUse)){
                    //which means the very last legal use of this resource is a cc.continue, if we deallocate this resource after cc.continue, the deallocation will be
                    //skipped by the control flow. return is irrelevant for quantum types
                    builder.setInsertionPoint(lastUse);
                }
                else{
                    builder.setInsertionPointAfter(lastUse);
                }
                cudaq::quake::DeallocOp::create(builder, lastUse->getLoc(), currentRoot);
            }
        }
    }
};

#endif
} // namespace

#ifdef RELEASE
CUDAQ_REGISTER_MLIR_PASS(LifetimeShortening)
#elif defined(DEBUG_PROVENANCE)
CUDAQ_REGISTER_MLIR_PASS(TestProvenancePass)
#endif