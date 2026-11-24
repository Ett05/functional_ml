module Main where

import Control.Exception (evaluate)
import Control.Monad (forM_)
import Data.List (sort)
import GHC.Clock (getMonotonicTimeNSec)
import GHC.Conc (getNumCapabilities)
import Text.Printf (printf)
import FunctionalML

-- | Force evaluation of Storable Vec results in O(1) without unpacking to a list
forceVec :: Vec -> Double
forceVec v
  | vecLength v == 0 = 0.0
  | otherwise        = indexVec v 0 + indexVec v (vecLength v - 1)

{-# NOINLINE runTapeBench #-}
runTapeBench :: [Double] -> [([Double], [Double])] -> Int -> Int -> Int -> IO Double
runTapeBench flatParams samples inDim hiddenDim outDim = do
  let res = map (\(x, y) -> valueAndGradN (\pExprs -> do
              inExprs <- mapM constExpr x
              preds   <- mlpForwardTape pExprs inExprs inDim hiddenDim outDim
              mseLossTape preds y
            ) flatParams) samples
  evaluate (sum (map fst res) + sum (map (head . snd) res))

{-# NOINLINE runStorable1T #-}
runStorable1T :: Vec -> Vec -> Vec -> Int -> Int -> Int -> Int -> IO Double
runStorable1T paramsVec xVec yVec batchSize inDim hiddenDim outDim = do
  let (lv, gv) = vmapMlpValueAndGrad paramsVec xVec yVec batchSize inDim hiddenDim outDim
  evaluate (forceVec lv + forceVec gv)

{-# NOINLINE runStorablePar #-}
runStorablePar :: Vec -> Vec -> Vec -> Int -> Int -> Int -> Int -> IO Double
runStorablePar paramsVec xVec yVec batchSize inDim hiddenDim outDim = do
  let (lv, gv) = vmapMlpValueAndGradPar paramsVec xVec yVec batchSize inDim hiddenDim outDim
  evaluate (forceVec lv + forceVec gv)

-- | Time an IO action in microseconds across `iters` runs (after `warmup` runs)
measureUs :: Int -> Int -> (Int -> IO Double) -> IO (Double, Double, Double)
measureUs warmup iters action = do
  forM_ [1 .. warmup] $ \i -> action i
  times <- mapM (\i -> do
    t0 <- getMonotonicTimeNSec
    !_ <- action i
    t1 <- getMonotonicTimeNSec
    pure (fromIntegral (t1 - t0) / 1000.0 :: Double)
    ) [1 .. iters]
  let sorted = sort times
      minT = head sorted
      medT = sorted !! (iters `div` 2)
      p95T = sorted !! min (iters - 1) ((iters * 95) `div` 100)
  pure (minT, medT, p95T)

main :: IO ()
main = do
  caps <- getNumCapabilities
  putStrLn "================================================================="
  putStrLn "  FunctionalML Benchmark: vmap(valueAndGrad) Per-Sample Kernel   "
  putStrLn "================================================================="
  printf "GHC Capabilities (threads): %d\n\n" caps

  -- Model architecture: 16 -> 64 -> 4
  let inDim = 16
      hiddenDim = 64
      outDim = 4
      mlp = initMLP inDim hiddenDim outDim
      flatParams = flattenMLP mlp
      numParams = length flatParams
      paramsVec = fromList flatParams

  -- -------------------------------------------------------------
  -- 1. Correctness Check: Fused C/NEON Kernel vs Pure Reverse AD
  -- -------------------------------------------------------------
  putStrLn "--- 1. Correctness Check (Storable C/NEON vs Reverse.hs Tape) ---"
  let checkBatch = 8
      sampleInputs  = [ [sin (fromIntegral (s * inDim + i) * 0.3) | i <- [0 .. inDim - 1]] | s <- [0 .. checkBatch - 1] ]
      sampleTargets = [ [cos (fromIntegral (s * outDim + o) * 0.5) | o <- [0 .. outDim - 1]] | s <- [0 .. checkBatch - 1] ]
      inVec  = fromList (concat sampleInputs)
      tgtVec = fromList (concat sampleTargets)

      (cLossesV, cGradsV) = vmapMlpValueAndGrad paramsVec inVec tgtVec checkBatch inDim hiddenDim outDim
      cLosses = toList cLossesV
      cGrads  = toList cGradsV

      hsPerSample (x, y) = valueAndGradN (\pExprs -> do
          inExprs <- mapM constExpr x
          preds   <- mlpForwardTape pExprs inExprs inDim hiddenDim outDim
          mseLossTape preds y
        ) flatParams
      hsResults = map hsPerSample (zip sampleInputs sampleTargets)
      hsLosses  = map fst hsResults
      hsGrads   = concatMap snd hsResults

      maxLossErr = maximum (zipWith (\a b -> abs (a - b)) cLosses hsLosses)
      maxGradErr = maximum (zipWith (\a b -> abs (a - b)) cGrads hsGrads)

  printf "Max Loss Diff: %.3e\n" maxLossErr
  printf "Max Grad Diff: %.3e\n" maxGradErr
  if maxLossErr < 1e-12 && maxGradErr < 1e-12
    then putStrLn "PASS: Fused C/NEON kernel matches pure Reverse AD to machine precision.\n"
    else error "FAIL: Numerical mismatch between C/NEON kernel and Reverse AD!"

  -- -------------------------------------------------------------
  -- 2. Scaling Benchmark across Batch Sizes
  -- -------------------------------------------------------------
  putStrLn "--- 2. Throughput & Latency Benchmark (MLP 16 -> 64 -> 4) ---"
  printf "Parameters per sample: %d\n\n" numParams
  printf "%-10s | %-23s | %-11s | %-11s | %-11s | %-12s\n"
    ("Batch" :: String) ("Backend" :: String) ("Min (us)" :: String) ("Med (us)" :: String) ("p95 (us)" :: String) ("Samples/sec" :: String)
  putStrLn (replicate 91 '-')

  forM_ [64, 256, 1024, 4096] $ \batchSize -> do
    let xs = [ sin (fromIntegral (s * inDim + i) * 0.17) | s <- [0 .. batchSize - 1], i <- [0 .. inDim - 1] ]
        ys = [ cos (fromIntegral (s * outDim + o) * 0.23) | s <- [0 .. batchSize - 1], o <- [0 .. outDim - 1] ]
        xVec = fromList xs
        yVec = fromList ys
        samples = [ (take inDim (drop (s * inDim) xs), take outDim (drop (s * outDim) ys)) | s <- [0 .. batchSize - 1] ]

    -- Only run slow Haskell list Tape on batch <= 256 so bench stays snappy
    if batchSize <= 256
      then do
        (mn, md, p95) <- measureUs 1 3 $ \_ ->
          runTapeBench flatParams samples inDim hiddenDim outDim
        let sps = fromIntegral batchSize / (md / 1e6)
        printf "%-10d | %-23s | %11.1f | %11.1f | %11.1f | %12.0f\n"
          batchSize ("Haskell Tape ([Double])" :: String) mn md p95 sps
      else pure ()

    -- Storable C/NEON Single-Core
    (mn1, md1, p95_1) <- measureUs 5 25 $ \_ ->
      runStorable1T paramsVec xVec yVec batchSize inDim hiddenDim outDim
    let sps1 = fromIntegral batchSize / (md1 / 1e6)
    printf "%-10d | %-23s | %11.1f | %11.1f | %11.1f | %12.0f\n"
      batchSize ("Storable C/NEON (1T)" :: String) mn1 md1 p95_1 sps1

    -- Storable C/NEON Multi-Core Parallel
    (mnP, mdP, p95_P) <- measureUs 5 25 $ \_ ->
      runStorablePar paramsVec xVec yVec batchSize inDim hiddenDim outDim
    let spsP = fromIntegral batchSize / (mdP / 1e6)
    printf "%-10d | %-23s | %11.1f | %11.1f | %11.1f | %12.0f\n"
      batchSize ("Storable C/NEON (Par)" :: String) mnP mdP p95_P spsP
    putStrLn (replicate 91 '-')
