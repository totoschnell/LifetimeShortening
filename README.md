# LifetimeShortening
This repo only contains the implementation of the pass, named as "lifetime-shortening". The compilation of this implementation must, however, rely on the local CUDA-Q build and LLVM/MLIR dependency path. The pass works now as a pass plugin of cudaq-opt. In order to insert this pass into the pipeline, several minor modifications are done to the current default pass pipeline:
1. unwind-lowering is replaced by unwind-by-dataflow, because no lowering is done by unwind-by-dataflow, thereby the IR still remains structured
2. combine-quantum-alloc is removed from the pipeline, because this pass is not necessary for the correctness, but it wipes out optimization done by lifetime-shortening
3. factor-quantum-alloc is removed from the pipeline, because at the moment there is a bug in memtoreg if this pass is applied together with lifetime-shortening
4. inject-implicit-output is removed from the pipeline, because this pass is used to guarantee the precondition of a very stupid design, which is completely useless, and this pass breaks the precondition of lifetime-shortening

lifetime-shortening is designed and tested between unwind-by-data and add-dealloc.