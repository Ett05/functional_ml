{-# LANGUAGE StrictData #-}
module FunctionalML.Core
  ( Vector
  , Matrix
  -- * Vector & Matrix Operations
  , dot
  , vecAdd
  , vecSub
  , scalarMul
  , outer
  , matVecMul
  , matMul
  , transpose
  -- * Activation Functions (Pure)
  , relu
  , reluPrime
  , sigmoid
  , sigmoidPrime
  , softmax
  ) where

import qualified Data.List as L

type Vector = [Double]
type Matrix = [[Double]]

-- | Dot product of two vectors
dot :: Vector -> Vector -> Double
dot u v = sum (zipWith (*) u v)

-- | Element-wise vector addition
vecAdd :: Vector -> Vector -> Vector
vecAdd = zipWith (+)

-- | Element-wise vector subtraction
vecSub :: Vector -> Vector -> Vector
vecSub = zipWith (-)

-- | Scalar multiplication
scalarMul :: Double -> Vector -> Vector
scalarMul s = map (* s)

-- | Outer product: u * v^T yielding a Matrix
outer :: Vector -> Vector -> Matrix
outer u v = [[ui * vj | vj <- v] | ui <- u]

-- | Matrix-Vector multiplication: M * v
matVecMul :: Matrix -> Vector -> Vector
matVecMul m v = map (`dot` v) m

-- | Matrix transpose
transpose :: Matrix -> Matrix
transpose = L.transpose

-- | Matrix-Matrix multiplication: A * B
matMul :: Matrix -> Matrix -> Matrix
matMul a b =
  let bT = transpose b
  in [[rowA `dot` colB | colB <- bT] | rowA <- a]

-- | Rectified Linear Unit (ReLU)
relu :: Double -> Double
relu x = max 0 x

-- | Derivative of ReLU
reluPrime :: Double -> Double
reluPrime x
  | x > 0     = 1.0
  | otherwise = 0.0

-- | Sigmoid activation
sigmoid :: Double -> Double
sigmoid x = 1.0 / (1.0 + exp (-x))

-- | Derivative of Sigmoid given its output y = sigmoid(x)
sigmoidPrime :: Double -> Double
sigmoidPrime y = y * (1.0 - y)

-- | Softmax over a vector (numerically stabilized by subtracting max)
softmax :: Vector -> Vector
softmax xs =
  let m = maximum xs
      exps = map (\x -> exp (x - m)) xs
      total = sum exps
  in map (/ total) exps
