{-# LANGUAGE StrictData #-}
module FunctionalML.Autograd.Reverse
  ( Expr (..)
  , Tape
  , TapeM
  , runTape
  , var
  , constExpr
  , addE
  , subE
  , mulE
  , divE
  , negE
  , expE
  , logE
  , reluE
  , sigmoidE
  , grad
  , grad2
  , gradN
  , valueAndGradN
  ) where

import Control.Monad.State.Strict
import qualified Data.IntMap.Strict as IM

-- | An Expression node holding its unique ID and evaluated value
data Expr = Expr
  { exprId  :: !Int
  , exprVal :: !Double
  } deriving (Eq, Show)

-- | A Node on the computation tape:
-- Stores the primal value and dependencies as (parentId, partial_derivative)
data Node = Node
  { nVal  :: !Double
  , nDeps :: ![(Int, Double)] -- Vector-Jacobian Product (VJP) components
  } deriving (Show)

-- | Computation Tape storing all recorded operations in creation order
data Tape = Tape
  { nextId :: !Int
  , nodes  :: !(IM.IntMap Node)
  }

emptyTape :: Tape
emptyTape = Tape 0 IM.empty

type TapeM = State Tape

-- | Allocate a new variable node on the tape
var :: Double -> TapeM Expr
var v = do
  t <- get
  let nid = nextId t
      node = Node v []
  put $! t { nextId = nid + 1, nodes = IM.insert nid node (nodes t) }
  pure (Expr nid v)

-- | Allocate a constant (no dependencies)
constExpr :: Double -> TapeM Expr
constExpr = var

-- | Record a binary operation on the tape
recordBinOp :: Double -> Double -> Double -> Expr -> Expr -> TapeM Expr
recordBinOp v dL dR (Expr idL _) (Expr idR _) = do
  t <- get
  let nid = nextId t
      node = Node v [(idL, dL), (idR, dR)]
  put $! t { nextId = nid + 1, nodes = IM.insert nid node (nodes t) }
  pure (Expr nid v)

-- | Record a unary operation on the tape
recordUnaryOp :: Double -> Double -> Expr -> TapeM Expr
recordUnaryOp v d (Expr idIn _) = do
  t <- get
  let nid = nextId t
      node = Node v [(idIn, d)]
  put $! t { nextId = nid + 1, nodes = IM.insert nid node (nodes t) }
  pure (Expr nid v)

-- Primitive differentiable operations
addE :: Expr -> Expr -> TapeM Expr
addE a b = recordBinOp (exprVal a + exprVal b) 1.0 1.0 a b

subE :: Expr -> Expr -> TapeM Expr
subE a b = recordBinOp (exprVal a - exprVal b) 1.0 (-1.0) a b

mulE :: Expr -> Expr -> TapeM Expr
mulE a b = recordBinOp (exprVal a * exprVal b) (exprVal b) (exprVal a) a b

divE :: Expr -> Expr -> TapeM Expr
divE a b =
  let ua = exprVal a
      ub = exprVal b
  in recordBinOp (ua / ub) (1.0 / ub) (- ua / (ub * ub)) a b

negE :: Expr -> TapeM Expr
negE a = recordUnaryOp (- exprVal a) (-1.0) a

expE :: Expr -> TapeM Expr
expE a =
  let ea = exp (exprVal a)
  in recordUnaryOp ea ea a

logE :: Expr -> TapeM Expr
logE a =
  let ua = exprVal a
  in recordUnaryOp (log ua) (1.0 / ua) a

reluE :: Expr -> TapeM Expr
reluE a =
  let ua = exprVal a
      v  = max 0 ua
      d  = if ua > 0 then 1.0 else 0.0
  in recordUnaryOp v d a

sigmoidE :: Expr -> TapeM Expr
sigmoidE a =
  let s = 1.0 / (1.0 + exp (- exprVal a))
      d = s * (1.0 - s)
  in recordUnaryOp s d a

-- | Run the Tape and evaluate reverse-mode automatic differentiation (Backprop).
-- By evaluating nodes in descending index order, we process the DAG in reverse topological order,
-- accumulating adjoints in linear O(N) time and cleanly avoiding the Diamond Problem.
runTape :: TapeM Expr -> (Double, IM.IntMap Double)
runTape computation =
  let (target, finalTape) = runState computation emptyTape
      totalNodes = nextId finalTape
      nodeMap = nodes finalTape
      -- Seed target loss with adjoint = 1.0
      initialGrads = IM.singleton (exprId target) 1.0
      -- Backpropagate down through tape
      backprop nid grads
        | nid < 0   = grads
        | otherwise =
            case IM.lookup nid grads of
              Nothing -> backprop (nid - 1) grads
              Just g  ->
                case IM.lookup nid nodeMap of
                  Nothing -> backprop (nid - 1) grads
                  Just (Node _ deps) ->
                    let updatedGrads = foldl (\acc (parentId, partial) ->
                                                IM.insertWith (+) parentId (g * partial) acc
                                             ) grads deps
                    in backprop (nid - 1) updatedGrads
      allGrads = backprop (totalNodes - 1) initialGrads
  in (exprVal target, allGrads)

-- | Compute gradient for a single-variable function: d f(x) / dx
grad :: (Expr -> TapeM Expr) -> Double -> Double
grad f x =
  let (_, grads) = runTape $ do
        xVar <- var x
        f xVar
  in IM.findWithDefault 0.0 0 grads

-- | Compute gradient for a two-variable function: (df/dx, df/dy)
grad2 :: (Expr -> Expr -> TapeM Expr) -> Double -> Double -> (Double, Double)
grad2 f x y =
  let (_, grads) = runTape $ do
        xVar <- var x
        yVar <- var y
        f xVar yVar
  in (IM.findWithDefault 0.0 0 grads, IM.findWithDefault 0.0 1 grads)

-- | Compute gradients for N variables
gradN :: ([Expr] -> TapeM Expr) -> [Double] -> [Double]
gradN f xs = snd (valueAndGradN f xs)

-- | Compute both primal value and gradients for N variables
valueAndGradN :: ([Expr] -> TapeM Expr) -> [Double] -> (Double, [Double])
valueAndGradN f xs =
  let (valOut, grads) = runTape $ do
        vars <- mapM var xs
        f vars
      gradList = [IM.findWithDefault 0.0 i grads | i <- [0 .. length xs - 1]]
  in (valOut, gradList)
