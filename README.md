# FunctionalML

A purely functional machine learning and automatic differentiation library written in **Haskell**.

This project demonstrates how core functional programming principles—**immutability, referential transparency, higher-order functions, and algebraic types**—can be used to build a robust, mathematically sound machine learning engine.

---

## Features

1. **Forward-Mode Automatic Differentiation (`Dual Numbers`)**
   - Implements `Dual a = Dual { val :: a, eps :: a }` where $\varepsilon^2 = 0$.
   - Full `Num`, `Fractional`, and `Floating` instances.
   - Computes exact derivatives in a single forward evaluation without numerical approximations.

2. **Reverse-Mode Automatic Differentiation (`TapeM` DAG Monad)**
   - Linear-time $O(N)$ backpropagation using Vector-Jacobian Products (VJPs).
   - Solves the **"Diamond Problem"** (graph sharing) by accumulating adjoints across the topological ordering of the DAG.
   - Supports multivariate gradient functions: `grad`, `grad2`, `gradN`, `valueAndGradN`.

3. **Pure Neural Network Layers & Architectures**
   - Explicit parameter separation: models are defined as transform functions over immutable record parameters (`DenseParams`, `MLPParams`).
   - Pure activation functions: `ReLU`, `Sigmoid`, `Softmax`.

4. **Pure State Transition Optimizers**
   - **SGD:** Pure parameter updates with no side effects.
   - **Adam:** Stateful optimizer represented as a pure state transition function:
     ```haskell
     adamStep :: AdamConfig -> AdamState -> [Double] -> [Double] -> (AdamState, [Double])
     ```

5. **Pure Training Loops**
   - Implemented via pure recursion, eliminating imperative `while` or `for` loops.

---

## Project Structure

```
functional_ml/
├── functional-ml.cabal            # Cabal build description (GHC 9.10+)
├── CONVERSATION.md                # Full discussion notes, theory, & FP advantages in ML
├── README.md                      # Project documentation
├── src/
│   ├── FunctionalML.hs            # Library entrypoint
│   ├── FunctionalML/
│   │   ├── Core.hs                # Vectors, matrices, pure linear algebra ops
│   │   ├── Autograd/
│   │   │   ├── Dual.hs            # Forward-mode AD with Dual numbers
│   │   │   └── Reverse.hs         # Reverse-mode AD with DAG tape
│   │   └── NN/
│   │       ├── Layers.hs          # Dense & MLP layers with explicit parameter records
│   │       ├── Loss.hs            # MSE and Binary Cross-Entropy losses
│   │       └── Optim.hs           # Pure SGD and Adam optimizers
└── app/
    └── Main.hs                    # Executable demo showcasing all components
```

---

## How to Build & Run

### Prerequisites
- **GHC** (tested on GHC 9.10.3)
- **Cabal** (tested on Cabal 3.16.1.0)

### Commands

To build the library and executable:
```bash
cabal build
```

To run the interactive demonstration:
```bash
cabal run
```

To explore the library in the GHCi REPL:
```bash
cabal repl
```

---

## Example Usage

### 1. Forward-Mode Differentiation with Dual Numbers
```haskell
import FunctionalML.Autograd.Dual

-- Define any mathematical function
f :: Dual Double -> Dual Double
f x = x^3 - 4*x + 1.0

-- Exact derivative at x = 3.0 (Expected: 3*(3^2) - 4 = 23.0)
df = diff f 3.0
```

### 2. Reverse-Mode Differentiation
```haskell
import FunctionalML.Autograd.Reverse

-- f(x, y) = x^2 * y + exp(x)
loss [x, y] = do
  xSq   <- mulE x x
  term1 <- mulE xSq y
  term2 <- expE x
  addE term1 term2

-- Compute value and gradients: (lossVal, [df/dx, df/dy])
(val, grads) = valueAndGradN loss [2.0, 3.0]
```

### 3. Pure Neural Network Forward Pass
```haskell
import FunctionalML

let mlp = initMLP 2 4 1
let output = mlpForward mlp [1.0, 0.0]
```

---

## Learning Objectives & Further Roadmap

For the full design discussion, comparison between Haskell and OCaml, and why functional programming is the backbone of modern frameworks like JAX, read **[CONVERSATION.md](file:///Users/giriksetya/Desktop/Coding/functional_ml/CONVERSATION.md)**.
