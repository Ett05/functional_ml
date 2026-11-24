module Main where

import Text.Printf (printf)
import Control.Monad (forM_, foldM)
import FunctionalML

main :: IO ()
main = do
  putStrLn "================================================================="
  putStrLn "          FunctionalML: Pure Functional Machine Learning         "
  putStrLn "=================================================================\n"

  -- -------------------------------------------------------------
  -- Demo 1: Forward-Mode AD (Dual Numbers)
  -- -------------------------------------------------------------
  putStrLn "--- Demo 1: Forward-Mode AD via Dual Numbers ---"
  let f :: Dual Double -> Dual Double
      f x = x * x * x - 4 * x + 1.0  -- f'(x) = 3x^2 - 4

      xVal :: Double
      xVal = 3.0

      valF :: Double
      valF  = val (f (Dual xVal 0.0))

      gradF :: Double
      gradF = diff f xVal

  printf "f(x)  = x^3 - 4x + 1\n"
  printf "f(%.1f)  = %.4f (Expected: 16.0)\n" xVal valF
  printf "f'(%.1f) = %.4f (Expected: 3*(3^2) - 4 = 23.0)\n\n" xVal gradF

  let g :: Dual Double -> Dual Double
      g x = sin x * exp x

      valG :: Double
      valG = val (g (Dual 1.0 0.0))

      gradG :: Double
      gradG = diff g 1.0

  printf "g(x)  = sin(x) * exp(x)\n"
  printf "g(1.0)  = %.4f\n" valG
  printf "g'(1.0) = %.4f (Analytical: cos(1)*exp(1) + sin(1)*exp(1) = 3.6558)\n\n" gradG

  -- -------------------------------------------------------------
  -- Demo 2: Reverse-Mode AD (Vector-Jacobian Products / DAG Tape)
  -- -------------------------------------------------------------
  putStrLn "--- Demo 2: Reverse-Mode AD via Directed Acyclic Graph Tape ---"
  -- f(x, y) = x^2 * y + exp(x)
  -- df/dx = 2*x*y + exp(x) = 2*(2)*(3) + exp(2) = 12 + 7.389 = 19.389
  -- df/dy = x^2 = 4.0
  let multivariateLoss :: [Expr] -> TapeM Expr
      multivariateLoss [x, y] = do
        xSq  <- mulE x x
        term1 <- mulE xSq y
        term2 <- expE x
        addE term1 term2
      multivariateLoss _ = error "Expected 2 inputs"

      (lossVal, gradList) = valueAndGradN multivariateLoss [2.0, 3.0]
      dfdx = case gradList of (dx:_) -> dx; [] -> 0.0
      dfdy = case drop 1 gradList of (dy:_) -> dy; [] -> 0.0

  printf "f(x, y) = x^2 * y + exp(x) evaluated at (x=2.0, y=3.0)\n"
  printf "f(2.0, 3.0)  = %.4f\n" lossVal
  printf "df/dx        = %.4f (Expected: ~19.3891)\n" dfdx
  printf "df/dy        = %.4f (Expected: 4.0000)\n\n" dfdy

  -- -------------------------------------------------------------
  -- Demo 3: Training a 2-Layer Neural Network on the XOR Problem
  -- -------------------------------------------------------------
  putStrLn "--- Demo 3: Pure Functional Neural Network Training (XOR Gate) ---"
  putStrLn "XOR is non-linearly separable and requires a hidden layer (2 -> 4 -> 1)."

  let dataset :: [([Double], [Double])]
      dataset =
        [ ([0.0, 0.0], [0.0])
        , ([0.0, 1.0], [1.0])
        , ([1.0, 0.0], [1.0])
        , ([1.0, 1.0], [0.0])
        ]

      inDim, hiddenDim, outDim :: Int
      inDim = 2
      hiddenDim = 4
      outDim = 1

      initialParams = initMLP inDim hiddenDim outDim
      initialFlat   = flattenMLP initialParams
      adamConfig    = defaultAdamConfig { alpha = 0.08 }
      initialAdam   = initAdamState (length initialFlat)

  putStrLn "\nInitial Predictions (before training):"
  forM_ dataset $ \(inputVec, targetVec) -> do
    let predVec = mlpForward initialParams inputVec
    printf "  Input: %s -> Target: %s, Prediction: %.4f\n"
      (show inputVec) (show targetVec) (case predVec of (p:_) -> p; [] -> 0.0)

  -- Loss function over the whole batch for a parameter vector
  let batchLoss :: [Expr] -> TapeM Expr
      batchLoss paramsFlat = do
        let forwardSample (xVec, yVec) = do
              inputExprs <- mapM constExpr xVec
              predExprs  <- mlpForwardTape paramsFlat inputExprs inDim hiddenDim outDim
              mseLossTape predExprs yVec

        sampleLosses <- mapM forwardSample dataset
        sumLosses    <- case sampleLosses of
                          []     -> constExpr 0.0
                          (l:ls) -> foldM addE l ls
        nConst       <- constExpr (fromIntegral (length dataset))
        divE sumLosses nConst

  -- Pure training loop using recursion (no while loops, no mutation)
  let trainEpoch :: ([Double], AdamState) -> (([Double], AdamState), Double)
      trainEpoch (params, optState) =
        let (loss, grads) = valueAndGradN batchLoss params
            (nextOptState, nextParams) = adamStep adamConfig optState params grads
        in ((nextParams, nextOptState), loss)

  let runTraining :: Int -> ([Double], AdamState) -> IO ([Double], AdamState)
      runTraining 0 st = pure st
      runTraining epochs (params, optState) = do
        let ((pNext, optNext), loss) = trainEpoch (params, optState)
        let currentEpoch = 300 - epochs
        if currentEpoch `mod` 50 == 0 || currentEpoch == 1
          then printf "Epoch %3d | Loss: %.6f\n" currentEpoch loss
          else pure ()
        runTraining (epochs - 1) (pNext, optNext)

  putStrLn "\nTraining purely via recursion & state transitions (300 Epochs):"
  (finalFlat, _) <- runTraining (300 :: Int) (initialFlat, initialAdam)

  let finalParams = unflattenMLP inDim hiddenDim outDim finalFlat
  putStrLn "\nFinal Predictions (after training):"
  forM_ dataset $ \(inputVec, targetVec) -> do
    let predVec = mlpForward finalParams inputVec
    printf "  Input: %s -> Target: %s, Prediction: %.4f\n"
      (show inputVec) (show targetVec) (case predVec of (p:_) -> p; [] -> 0.0)

  putStrLn "\nSuccess! Pure functional machine learning running seamlessly in Haskell."
