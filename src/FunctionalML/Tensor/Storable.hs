{-# LANGUAGE CApiFFI #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE StrictData #-}
module FunctionalML.Tensor.Storable
  ( Vec (..)
  , fromList
  , toList
  , replicateVec
  , sliceVec
  , vecLength
  , indexVec
  , dotStorable
  , matVecBiasStorable
  , vmapMlpValueAndGrad
  , vmapMlpValueAndGradPar
  ) where

import Control.Concurrent (forkIO, newEmptyMVar, putMVar, takeMVar)
import Control.Monad (forM, forM_)
import Foreign.C.Types (CDouble (..), CInt (..))
import Foreign.ForeignPtr (ForeignPtr, mallocForeignPtrArray, withForeignPtr)
import Foreign.Marshal.Array (peekArray, pokeArray)
import Foreign.Ptr (Ptr, plusPtr)
import Foreign.Storable (peekElemOff, sizeOf)
import GHC.Conc (getNumCapabilities)
import System.IO.Unsafe (unsafePerformIO)

-- | Contiguous, immutable, zero-copy sliceable Storable vector of Doubles.
-- Backed by pinned memory (ForeignPtr) so it can be passed directly to
-- C/NEON/Metal kernels via FFI without copying.
data Vec = Vec
  { vecPtr    :: !(ForeignPtr Double)
  , vecOffset :: !Int
  , vecLen    :: !Int
  }

instance Show Vec where
  show v = "Vec " ++ show (toList v)

instance Eq Vec where
  a == b = toList a == toList b

vecLength :: Vec -> Int
vecLength = vecLen

-- | O(1) element lookup without copying to a list.
indexVec :: Vec -> Int -> Double
indexVec v@(Vec _ _ n) i
  | i < 0 || i >= n = error "indexVec: index out of bounds"
  | otherwise = unsafePerformIO $ withVecPtr v (`peekElemOff` i)

-- | O(1) zero-copy slice of a Storable vector.
sliceVec :: Int -> Int -> Vec -> Vec
sliceVec off len (Vec fp baseOff baseLen)
  | off < 0 || len < 0 || off + len > baseLen =
      error "sliceVec: slice out of bounds"
  | otherwise = Vec fp (baseOff + off) len

withVecPtr :: Vec -> (Ptr Double -> IO a) -> IO a
withVecPtr (Vec fp off _) action =
  withForeignPtr fp $ \p ->
    action (p `plusPtr` (off * sizeOf (0.0 :: Double)))

fromList :: [Double] -> Vec
fromList xs = unsafePerformIO $ do
  let n = length xs
  fp <- mallocForeignPtrArray n
  withForeignPtr fp (`pokeArray` xs)
  pure (Vec fp 0 n)

toList :: Vec -> [Double]
toList v@(Vec _ _ n) = unsafePerformIO $
  withVecPtr v (peekArray n)

replicateVec :: Int -> Double -> Vec
replicateVec n val = fromList (replicate n val)

-- ---------------------------------------------------------------------------
-- FFI Imports (C / ARM64 NEON Kernels)
-- ---------------------------------------------------------------------------

foreign import ccall unsafe "kernels.h fml_dot_f64"
  c_dot_f64 :: Ptr CDouble -> Ptr CDouble -> CInt -> IO CDouble

foreign import ccall unsafe "kernels.h fml_matvec_bias_f64"
  c_matvec_bias_f64 :: Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CInt -> CInt -> IO ()

foreign import ccall unsafe "kernels.h fml_vmap_mlp_value_and_grad_f64"
  c_vmap_mlp_value_and_grad_f64
    :: Ptr CDouble -> Ptr CDouble -> Ptr CDouble
    -> Ptr CDouble -> Ptr CDouble
    -> CInt -> CInt -> CInt -> CInt -> IO ()

-- ---------------------------------------------------------------------------
-- Pure High-Level Wrappers
-- ---------------------------------------------------------------------------

dotStorable :: Vec -> Vec -> Double
dotStorable a b
  | vecLen a /= vecLen b = error "dotStorable: length mismatch"
  | otherwise = unsafePerformIO $
      withVecPtr a $ \pa ->
      withVecPtr b $ \pb -> do
        CDouble res <- c_dot_f64 (castPtrDouble pa) (castPtrDouble pb) (fromIntegral (vecLen a))
        pure res

matVecBiasStorable :: Vec -> Vec -> Vec -> Int -> Int -> Vec
matVecBiasStorable w x b outDim inDim = unsafePerformIO $ do
  outFp <- mallocForeignPtrArray outDim
  withVecPtr w $ \pw ->
    withVecPtr x $ \px ->
    withVecPtr b $ \pb ->
    withForeignPtr outFp $ \pout ->
      c_matvec_bias_f64 (castPtrDouble pw) (castPtrDouble px) (castPtrDouble pb) (castPtrDouble pout)
        (fromIntegral outDim) (fromIntegral inDim)
  pure (Vec outFp 0 outDim)

-- | Single-threaded fused per-sample valueAndGrad over a contiguous batch.
-- Returns (lossesVec [batchSize], gradsVec [batchSize * numParams]).
vmapMlpValueAndGrad
  :: Vec -> Vec -> Vec -> Int -> Int -> Int -> Int -> (Vec, Vec)
vmapMlpValueAndGrad params inputs targets batchSize inDim hiddenDim outDim =
  unsafePerformIO $ do
    let numParams = hiddenDim * inDim + hiddenDim + outDim * hiddenDim + outDim
    lossesFp <- mallocForeignPtrArray batchSize
    gradsFp  <- mallocForeignPtrArray (batchSize * numParams)
    withVecPtr params  $ \pp ->
      withVecPtr inputs  $ \pi_ ->
      withVecPtr targets $ \pt ->
      withForeignPtr lossesFp $ \pl ->
      withForeignPtr gradsFp  $ \pg ->
        c_vmap_mlp_value_and_grad_f64
          (castPtrDouble pp) (castPtrDouble pi_) (castPtrDouble pt)
          (castPtrDouble pl) (castPtrDouble pg)
          (fromIntegral batchSize) (fromIntegral inDim)
          (fromIntegral hiddenDim) (fromIntegral outDim)
    pure (Vec lossesFp 0 batchSize, Vec gradsFp 0 (batchSize * numParams))

-- | Multi-core parallel fused per-sample valueAndGrad across GHC capabilities.
-- Because each sample writes to a disjoint slice of the output buffers,
-- parallel chunks execute with zero locks and zero data races.
vmapMlpValueAndGradPar
  :: Vec -> Vec -> Vec -> Int -> Int -> Int -> Int -> (Vec, Vec)
vmapMlpValueAndGradPar params inputs targets batchSize inDim hiddenDim outDim =
  unsafePerformIO $ do
    caps <- getNumCapabilities
    if caps <= 1 || batchSize < caps * 16
      then pure (vmapMlpValueAndGrad params inputs targets batchSize inDim hiddenDim outDim)
      else do
        let numParams = hiddenDim * inDim + hiddenDim + outDim * hiddenDim + outDim
        lossesFp <- mallocForeignPtrArray batchSize
        gradsFp  <- mallocForeignPtrArray (batchSize * numParams)
        let chunkSize = (batchSize + caps - 1) `div` caps
            ranges =
              [ (start, min chunkSize (batchSize - start))
              | c <- [0 .. caps - 1]
              , let start = c * chunkSize
              , start < batchSize
              ]
        dones <- forM ranges $ \(start, count) -> do
          done <- newEmptyMVar
          _ <- forkIO $ do
            withVecPtr params  $ \pp ->
              withVecPtr inputs  $ \pi_ ->
              withVecPtr targets $ \pt ->
              withForeignPtr lossesFp $ \pl ->
              withForeignPtr gradsFp  $ \pg ->
                c_vmap_mlp_value_and_grad_f64
                  (castPtrDouble pp)
                  (castPtrDouble (pi_ `plusPtr` (start * inDim * sizeD)))
                  (castPtrDouble (pt `plusPtr` (start * outDim * sizeD)))
                  (castPtrDouble (pl `plusPtr` (start * sizeD)))
                  (castPtrDouble (pg `plusPtr` (start * numParams * sizeD)))
                  (fromIntegral count) (fromIntegral inDim)
                  (fromIntegral hiddenDim) (fromIntegral outDim)
            putMVar done ()
          pure done
        forM_ dones takeMVar
        pure (Vec lossesFp 0 batchSize, Vec gradsFp 0 (batchSize * numParams))
  where
    sizeD = sizeOf (0.0 :: Double)

castPtrDouble :: Ptr Double -> Ptr CDouble
castPtrDouble p = p `plusPtr` 0
