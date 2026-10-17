#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Analysis/SliceWalk.h"

//#define DEBUG_PROVENANCE
#ifdef DEBUG_PROVENANCE
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"
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

//general alias analysis for memory-semantic quake IR
class AA {
private:
    //the refs in the set are may-alias with the key, because they are derived from the key
    DenseMap<Node, DenseSet<Node>> deriveGraph;
    //root references to quantum resources which is allocated inside this kernel
    DenseSet<Value> internalProvenanceRoots;
    //root references to quantum resources which is passed to this kernel as arguments
    DenseSet<Value> externalProvenanceRoots;
    
    static bool isQuantumReferenceType(Type ty){
        return isa<cudaq::quake::RefType, cudaq::quake::VeqType, cudaq::quake::StruqType>(ty);
    }

    //struq reference definition coming from the control flow, from block arguments or from results of cc.if or cc.loop, track the provenance of the members
    void handleStruqFromControlFlow(Value &derivedStruq, Value &sourceStruq){
        auto struqType = dyn_cast<cudaq::quake::StruqType>(derivedStruq.getType());
        for(unsigned index = 0; index < struqType.getNumMembers(); ++index){
            deriveGraph[Node(sourceStruq, index)].insert(Node(derivedStruq, index));
        }
    }

    //process quantum reference from op results
    void processOpResult(Value &v, Operation *op){
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
                deriveGraph[Node(member, WHOLE)].insert(Node(v, WHOLE));
                deriveGraph[Node(member, WHOLE)].insert(Node(v, memberIdx++));
            }
        }
        else if(auto getMember = dyn_cast<cudaq::quake::GetMemberOp>(op)){
            Value struq = getMember.getStruq();
            uint32_t index = getMember.getIndex();
            deriveGraph[Node(struq, index)].insert(Node(v, WHOLE));
        }
        else{
            std::optional<llvm::SmallVector<Value>> predecessors =  getControlFlowPredecessors(v);
            if(!predecessors.has_value()){
                existRefWithoutReasonableProvenance = true;
                op->emitError("Detected quantum reference as op result without proper predecessors").attachNote(v.getLoc());
                deriveGraph.clear();
                internalProvenanceRoots.clear();
                externalProvenanceRoots.clear();
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
    }

    //process quantum reference from block arguments
    void processBlockArgument(BlockArgument blockArgument){
        std::optional<llvm::SmallVector<Value>> predecessors = getControlFlowPredecessors(blockArgument);
        if(!predecessors.has_value()){
            Block *block = blockArgument.getOwner();
            //function arguments
            if(block->isEntryBlock() && isa<func::FuncOp>(block->getParentOp())){
                externalProvenanceRoots.insert(blockArgument);
                if(auto struqType = dyn_cast<cudaq::quake::StruqType>(blockArgument.getType())){
                    for(unsigned index = 0; index < struqType.getNumMembers(); ++index){
                        deriveGraph[Node(blockArgument, WHOLE)].insert(Node(blockArgument, index));
                    }
                }
            }
            else{
                existRefWithoutReasonableProvenance = true;
                blockArgument.getOwner()->getParent()->getParentOp()->emitError("Detected quantum reference as block argument without reasonable provenance").attachNote(blockArgument.getLoc());
                deriveGraph.clear();
                internalProvenanceRoots.clear();
                externalProvenanceRoots.clear();
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
                    processOpResult(v, op);
                    if(existRefWithoutReasonableProvenance){
                        return WalkResult::interrupt();
                    }
                }
            }
            for(Region &r : op->getRegions()){
                for(Block &b : r.getBlocks()){
                    for(BlockArgument blockArgument : b.getArguments()){
                        if(isQuantumReferenceType(blockArgument.getType())){
                            processBlockArgument(blockArgument);
                            if(existRefWithoutReasonableProvenance){
                                return WalkResult::interrupt();
                            }
                        }
                    }
                }
            }
            return WalkResult::advance();
        });
    }

    //BFS deriveGraph for internal roots
    void buildInternalProvenance(void){
        if(existRefWithoutReasonableProvenance){
            return;
        }
        for(Value internalRoot : internalProvenanceRoots){
            internalRootProvenanceAnalysisResult[internalRoot];
            SetVector<Node> workingSet;
            //the root is an alias of itself
            workingSet.insert(Node(internalRoot, WHOLE));
            size_t head = 0;
            while(head != workingSet.size()){
                Node current = workingSet[head++];
                if(current.second == WHOLE){
                    internalRootProvenanceAnalysisResult[internalRoot].insert(current.first);
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

    //BFS deriveGraph for external roots
    void buildExternalProvenance(void){
        if(existRefWithoutReasonableProvenance){
            return;
        }
        for(Value externalRoot : externalProvenanceRoots){
            externalRootProvenanceAnalysisResult[externalRoot];
            SetVector<Node> workingSet;
            //the root is an alias of itself
            workingSet.insert(Node(externalRoot, WHOLE));
            size_t head = 0;
            while(head != workingSet.size()){
                Node current = workingSet[head++];
                if(current.second == WHOLE){
                    externalRootProvenanceAnalysisResult[externalRoot].insert(current.first);
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
    bool existRefWithoutReasonableProvenance;
    DenseMap<Value, DenseSet<Value>> internalRootProvenanceAnalysisResult;
    DenseMap<Value, DenseSet<Value>> externalRootProvenanceAnalysisResult;
    DenseMap<Value, DenseSet<Value>> mayAliasResult;//TODO: future task
    AA(func::FuncOp func){
        existRefWithoutReasonableProvenance = false;
        buildDeriveGraph(func);
        buildInternalProvenance();
        buildExternalProvenance();
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
                        //use out of define scope
                        malformedIR = true;
                        user->emitError("Detected dangling reference use").attachNote(alias.getLoc());
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
                LivingRange aliasLivingRange = findOrCreateAliasLivingRangeCache(alias);
                if(malformedIR){
                    return;
                }
                if(aliasLivingRange.has_value()){
                    Operation *firstUseInRootBlock = currentRootBlock->findAncestorOpInBlock(*(aliasLivingRange.value().first));
                    Operation *lastUseInRootBlock = currentRootBlock->findAncestorOpInBlock(*(aliasLivingRange.value().second));
                    if(!firstUseInRootBlock || !lastUseInRootBlock){
                        //here a dangling quantum reference is found. This is an undefined behavior. Report this in console. 
                        detectedDanglingReference = true;
                        Operation *escapingUse = !firstUseInRootBlock ? aliasLivingRange.value().first : aliasLivingRange.value().second;
                        escapingUse->emitError("detected dangling reference use").attachNote(currentRoot.getLoc()) << "quantum resource originates here";
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


//Test pass for provenance analysis. Generated by Chatgpt.
#ifdef DEBUG_PROVENANCE
struct TestProvenancePass
    : public PassWrapper<TestProvenancePass,
                         OperationPass<func::FuncOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestProvenancePass)

    StringRef getArgument() const override {
        return "test-quake-provenance";
    }

    StringRef getDescription() const override {
        return "Print quantum allocation provenance analysis";
    }

    void runOnOperation() override {
        func::FuncOp func = getOperation();

        AA analysis(func);

        llvm::raw_ostream &os = llvm::outs();

        os << "\n=== Provenance Analysis: "
           << func.getSymName() << " ===\n";

        if (analysis.existRefWithoutReasonableProvenance) {
            os << "ERROR: Unknown quantum provenance\n";
            signalPassFailure();
            return;
        }

        // All SSA values use the same naming context.
        AsmState asmState(func.getOperation());

        const DenseMap<Value, DenseSet<Value>> &result =
            analysis.internalRootProvenanceAnalysisResult;

        for (DenseMap<Value, DenseSet<Value>>::const_iterator it =
                 result.begin();
             it != result.end(); ++it) {

            os << "\nRoot: ";
            it->first.printAsOperand(os, asmState);
            os << "\n";

            os << "Provenance: {\n";

            for (Value ref : it->second) {
                os << "  ";
                ref.printAsOperand(os, asmState);
                os << " : ";
                ref.getType().print(os);
                os << "\n";
            }

            os << "}\n";
        }

        os << "\n=== End Provenance Analysis ===\n";
        os.flush();
    }
};
//end of code generated by Chatgpt

// Register the test pass.
static PassRegistration<TestProvenancePass> testProvenancePass;
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

    //the correctness of lifetime analysis relies on a precondition: The IR must only consist of structured ops. This implies that there is only one block 
    //in every region, because cc dialect has no op to br from one block to another. There should be no existing deallocs in the IR. If unwind-by-dataflow 
    //is used to replace unwind-lowering, and this pass is placed in front of add-deallocs, the precondition should hold for IRs generated by compiler front end.
    bool checkLifetimeShorteningPrecondition(){
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

    //currently we are not keeping the allocation order of quantum resources according to the definition order in the source file.
    void runOnOperation() override{
        if(!checkLifetimeShorteningPrecondition()){
            return;
        }
        func::FuncOp func = getOperation();
        AA aaResult(func);
        if(aaResult.existRefWithoutReasonableProvenance){
            signalPassFailure();
            return;
        }
        Lifetime lifetimeResult(aaResult.internalRootProvenanceAnalysisResult);
        if(lifetimeResult.malformedIR || lifetimeResult.detectedDanglingReference){
            signalPassFailure();
            return;
        }
        for(DenseMap<Value, LivingRange>::const_iterator it = lifetimeResult.internalRootLivingRange.begin(); it != lifetimeResult.internalRootLivingRange.end(); ++it){
            Value currentRoot = it->first;
            LivingRange currentRootLivingRange = it->second;
            Operation *definingOp = currentRoot.getDefiningOp();
            if(!currentRootLivingRange.has_value()){
                if(auto initStateOp = dyn_cast<cudaq::quake::InitializeStateOp>(definingOp)){
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
                    definingOp->moveBefore(firstUse);
                }
                OpBuilder builder(lastUse->getContext());
                builder.setInsertionPointAfter(lastUse);
                cudaq::quake::DeallocOp dealloc = cudaq::quake::DeallocOp::create(builder, lastUse->getLoc(), currentRoot);
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