# FunctionalML — Project Context, Status & Benchmarking Guide

This document provides the complete technical context of the `functional_ml` repository: what has been implemented, what remains to be built, and how to benchmark the system.

---

## 1. Project Mission & Core Philosophy

`FunctionalML` is a purely functional machine learning and automatic differentiation library written in **Haskell** (GHC 9.10+). Its goal is to demonstrate how functional programming principles—**immutability, referential transparency, higher-order functions, and algebraic data types**—map directly to modern ML systems (similar to Google's JAX, Flax, and Equinox), without hidden mutable state or side effects.

---

## 2. Repository Layout & Module Responsibilities

```
functional_ml/
├── functional-ml.cabal            # Cabal 3.0 package (GHC2021, base, containers, mtl, array)
├── README.md                      # Quickstart, features, and usage examples
├── CONVERSATION.md                # Theoretical discussion: FP in ML, Haskell vs. OCaml, AD theory
├── PROJECT_CONTEXT.md             # This file: architecture, status, roadmap, and benchmarking
├── app/
│   └── Main.hs                    # Runnable demos: Forward AD, Reverse AD, XOR MLP training
└── src/
    ├── FunctionalML.hs            # Top-level module re-exporting all public APIs
    └── FunctionalML/
        ├── Core.hs                # Pure linear algebra & activations (Vector = [Double], Matrix = [[Double]])
        ├── Autograd/
        │   ├── Dual.hs            # Forward-mode AD via Dual numbers (a + bε, ε² = 0)
        │   └── Reverse.hs         # Reverse-mode AD via State Tape DAG & VJP accumulation
        └── NN/
            ├── Layers.hs          # Immutable parameter records (DenseParams, MLPParams) & forward passes
            ├── Loss.hs            # Numerical & differentiable (TapeM) MSE and BCE losses
            └── Optim.hs           # Pure state-transition optimizers (sgdStep, adamStep)
```

### Key Implemented Details
1. **Forward-Mode AD (`FunctionalML.Autograd.Dual`)**:
   * `data Dual a = Dual { val :: a, eps :: a }` with `Num`, `Fractional`, and `Floating` instances.
   * `diff :: (Dual Double -> Dual Double) -> Double -> Double` computes exact derivatives in a single forward pass.
2. **Reverse-Mode AD (`FunctionalML.Autograd.Reverse`)**:
   * `TapeM = State Tape` records `Expr` nodes into an `IM.IntMap Node` with `(parentId, partialDerivative)` dependencies.
   * `runTape` sweeps node IDs in descending order (reverse topological order) to accumulate adjoints in **$O(N)$ linear time**, avoiding the exponential "Diamond Problem" when variables are reused.
   * Exports `grad`, `grad2`, `gradN`, and `valueAndGradN`.
3. **Pure Neural Networks & Optimizers (`FunctionalML.NN.*`)**:
   * Models separate immutable parameter records (`DenseParams`, `MLPParams`) from pure forward functions (`denseForward`, `mlpForward`, `mlpForwardTape`).
   * `flattenMLP` / `unflattenMLP` bridge structured records and flat `[Double]` parameter vectors for `valueAndGradN`.
   * Optimizers (`sgdStep`, `adamStep`) are pure state transitions returning `(nextState, nextParams)`.
   * `app/Main.hs` runs a tail-recursive 300-epoch training loop solving the non-linear XOR gate.

---

## 3. Current Status

### Completed
- [x] Cabal project scaffolding (`functional-ml.cabal`) with zero external network dependencies.
- [x] Pure linear algebra and activation functions (`Core.hs`).
- [x] Forward-mode automatic differentiation with Dual numbers (`Dual.hs`).
- [x] Reverse-mode automatic differentiation with $O(N)$ DAG tape (`Reverse.hs`).
- [x] Dense and 2-layer MLP layers with explicit parameter separation (`Layers.hs`).
- [x] MSE and Binary Cross-Entropy losses (`Loss.hs`).
- [x] Pure SGD and Adam optimizers (`Optim.hs`).
- [x] End-to-end runnable demo (`app/Main.hs`) verified via `cabal build` and `cabal run`.
- [x] Conceptual & design documentation (`README.md`, `CONVERSATION.md`, `PROJECT_CONTEXT.md`).

### Left To Do (Roadmap)
- [ ] **Contiguous Zero-Copy Memory (`Data.Vector.Storable`):** Replace linked lists (`[Double]`, `[[Double]]`) with `Storable` vectors so Haskell can pass raw `Ptr Double` / `Ptr Float` directly to custom kernels via FFI with zero copying.
- [ ] **Fused Per-Sample Gradient Kernel (`vmap(valueAndGrad)`):** Compile/execute fused forward + backward passes in a single stateless C/SIMD (NEON on Apple Silicon) or Metal kernel, parallelized across samples via Haskell with zero tape allocation and zero data races.
- [ ] **Head-to-Head PyTorch Benchmark Suite:** Benchmark `FunctionalML` (`tasty-bench`) against PyTorch (`torch.func.vmap(torch.func.grad(...))`, eager and `torch.compile`) on per-sample gradients and higher-order AD.
- [ ] **Generic N-Dimensional Tensor:** Generalize 1D/2D operations to arbitrary-rank `Tensor` types with shapes and strides.
- [ ] **Compile-Time Shape Safety:** Encode tensor dimensions in the type system using `DataKinds`, `GADTs`, and `TypeFamilies` (e.g. `matMul :: Tensor '[m, k] a -> Tensor '[k, n] a -> Tensor '[m, n] a`) so shape mismatches fail at compile time.
- [ ] **PyTree / Generic Parameter Traversal:** Replace manual `flattenMLP` / `unflattenMLP` with a `PyTree` / `Functor`-like `treeMap` abstraction over arbitrary nested parameter records.
- [ ] **Property-Based Testing (`QuickCheck`):** Verify `Dual` and `Reverse` gradients against numerical finite differences:
  $$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$
  and algebraic laws (linearity, product rule, chain rule).

---

## 4. Benchmarking Strategy & PyTorch Target

### Primary Competitive Target: `vmap(grad(loss))` (Per-Sample Gradients & Higher-Order AD)
Rather than competing on large dense `MatMul` (where PyTorch delegates to vendor assembly like Apple Accelerate / MKL / cuBLAS), `FunctionalML` targets workloads where **FP composability + fused custom kernels** outperform PyTorch structurally—even with `torch.compile`:
1. **Per-Sample Gradients (`vmap(valueAndGrad loss)`):** Core bottleneck in Differential Privacy (DP-SGD) and meta-learning. PyTorch relies on `torch.func.vmap(torch.func.grad(...))`, incurring graph-tracing overhead and memory expansion; pure FP composes `vmap . valueAndGrad` into a single parallel, tapeless fused kernel.
2. **Kernel Fusion & Zero Dispatch Overhead:** Pure expressions fuse forward + backward chains into a single memory pass in C/NEON/Metal with near-zero dispatch overhead compared to PyTorch's per-op Python/C++ dispatcher.

### Execution Pipeline
`Pure Haskell EDSL / Layers` → `Fused Forward + Backward Kernel (C / NEON / Metal)` → `Parallel Execution via Haskell FFI (Storable Ptr)`

### Internal Micro-Benchmarks (`tasty-bench` / `criterion` + GHC RTS)
1. **Forward-Mode vs. Reverse-Mode Scaling ($f : \mathbb{R}^N \to \mathbb{R}$):**
   * Scale input dimension $N \in \{10, 50, 100, 500, 1000\}$.
   * Forward AD (`diff` per input) scales as $O(N)$ passes; Reverse AD (`valueAndGradN`) computes all $N$ partial derivatives in $O(1)$ backward pass.
2. **Diamond Problem / Graph Sharing Stress Test:**
   * Evaluate $x_0 = x$, $x_{k+1} = x_k \cdot x_k$ for depth $K = 20$.
   * Verify runtime scales linearly ($20$ steps) rather than exponentially ($2^{20}$).
3. **Training Throughput & Memory Allocation:**
   * Run with GHC RTS statistics enabled to inspect total bytes allocated and GC pause overhead (crucial for measuring the speedup from `[Double]` to `Data.Vector.Storable`):
     ```bash
     cabal run -- +RTS -s
     ```
   * For time/allocation profiling per function:
     ```bash
     cabal run --enable-profiling -- +RTS -p
     ```
