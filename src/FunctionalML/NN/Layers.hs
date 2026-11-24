{-# LANGUAGE StrictData #-}
module FunctionalML.NN.Layers
  ( DenseParams (..)
  , MLPParams (..)
  , initDense
  , initMLP
  , denseForward
  , denseForwardTape
  , mlpForward
  , mlpForwardTape
  , flattenMLP
  , unflattenMLP
  ) where

import Control.Monad (zipWithM, foldM)
import FunctionalML.Core (Vector, matVecMul, relu, sigmoid, vecAdd)
import FunctionalML.Autograd.Reverse

-- | Parameters for a single Dense (Linear) Layer: y = W * x + b
data DenseParams = DenseParams
  { weights :: ![[Double]] -- shape: [outDim, inDim]
  , biases  :: ![Double]   -- shape: [outDim]
  } deriving (Show, Eq)

-- | Parameters for a 2-layer Multi-Layer Perceptron (MLP)
data MLPParams = MLPParams
  { layer1 :: !DenseParams
  , layer2 :: !DenseParams
  } deriving (Show, Eq)

-- | Initialize Dense layer parameters with simple deterministic weights
initDense :: Int -> Int -> Double -> DenseParams
initDense inDim outDim seed =
  let ws = [[ sin (fromIntegral (i * inDim + j) * seed + 0.1) * 0.5
            | j <- [1 .. inDim] ]
           | i <- [1 .. outDim] ]
      bs = [0.0 | _ <- [1 .. outDim]]
  in DenseParams ws bs

-- | Initialize a 2-layer MLP (inDim -> hiddenDim -> outDim)
initMLP :: Int -> Int -> Int -> MLPParams
initMLP inDim hiddenDim outDim =
  MLPParams
    { layer1 = initDense inDim hiddenDim 0.42
    , layer2 = initDense hiddenDim outDim 0.84
    }

-- | Pure forward pass for a Dense layer (using standard Doubles)
denseForward :: DenseParams -> Vector -> Vector
denseForward (DenseParams ws bs) x = vecAdd (matVecMul ws x) bs

-- | Pure forward pass for a 2-layer MLP with ReLU hidden and Sigmoid output
mlpForward :: MLPParams -> Vector -> Vector
mlpForward (MLPParams l1 l2) x =
  let h  = map relu (denseForward l1 x)
      o  = map sigmoid (denseForward l2 h)
  in o

-- | Differentiable forward pass for Dense layer inside the TapeM monad
denseForwardTape :: [Expr] -> [Expr] -> [Expr] -> Int -> Int -> TapeM [Expr]
denseForwardTape weightExprs biasExprs inputExprs inDim _outDim = do
  -- Split weightExprs into rows of length inDim
  let rows = chunk inDim weightExprs
  mapM (\(row, b) -> do
          terms <- zipWithM mulE row inputExprs
          sumTerms <- foldM addE b terms
          pure sumTerms
       ) (zip rows biasExprs)
  where
    chunk _ [] = []
    chunk n xs = take n xs : chunk n (drop n xs)

-- | Differentiable forward pass for the 2-layer MLP
mlpForwardTape :: [Expr] -> [Expr] -> Int -> Int -> Int -> TapeM [Expr]
mlpForwardTape paramExprs inputExprs inDim hiddenDim outDim = do
  let l1WeightCount = hiddenDim * inDim
      l1BiasCount   = hiddenDim
      l2WeightCount = outDim * hiddenDim
      l2BiasCount   = outDim

      (l1W, rest1)  = splitAt l1WeightCount paramExprs
      (l1B, rest2)  = splitAt l1BiasCount rest1
      (l2W, rest3)  = splitAt l2WeightCount rest2
      l2B           = take l2BiasCount rest3

  -- Layer 1 + ReLU
  hPre <- denseForwardTape l1W l1B inputExprs inDim hiddenDim
  hAct <- mapM reluE hPre

  -- Layer 2 + Sigmoid
  oPre <- denseForwardTape l2W l2B hAct hiddenDim outDim
  mapM sigmoidE oPre

-- | Flatten all MLP parameters into a flat list of Doubles for gradient computation
flattenMLP :: MLPParams -> [Double]
flattenMLP (MLPParams (DenseParams w1 b1) (DenseParams w2 b2)) =
  concat w1 ++ b1 ++ concat w2 ++ b2

-- | Reconstruct MLP parameters from a flat list of Doubles
unflattenMLP :: Int -> Int -> Int -> [Double] -> MLPParams
unflattenMLP inDim hiddenDim outDim flat =
  let l1WCount = hiddenDim * inDim
      l1BCount = hiddenDim
      l2WCount = outDim * hiddenDim
      l2BCount = outDim

      (w1F, r1) = splitAt l1WCount flat
      (b1F, r2) = splitAt l1BCount r1
      (w2F, r3) = splitAt l2WCount r2
      b2F       = take l2BCount r3

      unflattenMatrix rows cols xs =
        [take cols (drop (i * cols) xs) | i <- [0 .. rows - 1]]

      l1 = DenseParams (unflattenMatrix hiddenDim inDim w1F) b1F
      l2 = DenseParams (unflattenMatrix outDim hiddenDim w2F) b2F
  in MLPParams l1 l2
