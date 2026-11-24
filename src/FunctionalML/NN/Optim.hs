{-# LANGUAGE StrictData #-}
module FunctionalML.NN.Optim
  ( sgdStep
  , AdamConfig (..)
  , AdamState (..)
  , defaultAdamConfig
  , initAdamState
  , adamStep
  ) where

-- | Pure Stochastic Gradient Descent (SGD) step:
-- returns updated parameters with no mutation.
sgdStep :: Double -> [Double] -> [Double] -> [Double]
sgdStep lr = zipWith (\w g -> w - lr * g)

-- | Configuration parameters for the Adam optimizer
data AdamConfig = AdamConfig
  { alpha   :: !Double -- Learning rate (default: 0.001)
  , beta1   :: !Double -- Exponential decay rate for 1st moment (default: 0.9)
  , beta2   :: !Double -- Exponential decay rate for 2nd moment (default: 0.999)
  , epsilon :: !Double -- Small constant for numerical stability (default: 1e-8)
  } deriving (Show, Eq)

defaultAdamConfig :: AdamConfig
defaultAdamConfig = AdamConfig
  { alpha   = 0.01
  , beta1   = 0.9
  , beta2   = 0.999
  , epsilon = 1e-8
  }

-- | Pure state for the Adam optimizer
data AdamState = AdamState
  { tStep :: !Int
  , mVec  :: ![Double] -- First moment vector
  , vVec  :: ![Double] -- Second moment vector
  } deriving (Show, Eq)

initAdamState :: Int -> AdamState
initAdamState numParams = AdamState
  { tStep = 0
  , mVec  = replicate numParams 0.0
  , vVec  = replicate numParams 0.0
  }

-- | Pure Adam optimization step:
-- Takes (Config, CurrentState, Params, Gradients) -> (NewState, UpdatedParams)
-- Completely pure, deterministic, and free of hidden global state.
adamStep :: AdamConfig -> AdamState -> [Double] -> [Double] -> (AdamState, [Double])
adamStep cfg state params grads =
  let t = tStep state + 1
      b1 = beta1 cfg
      b2 = beta2 cfg
      lr = alpha cfg
      eps = epsilon cfg

      -- Update biased 1st and 2nd moment estimates
      mNew = zipWith (\m g -> b1 * m + (1.0 - b1) * g) (mVec state) grads
      vNew = zipWith (\v g -> b2 * v + (1.0 - b2) * (g * g)) (vVec state) grads

      -- Compute bias-corrected estimates
      b1Correction = 1.0 - (b1 ^ t)
      b2Correction = 1.0 - (b2 ^ t)

      mHat = map (/ b1Correction) mNew
      vHat = map (/ b2Correction) vNew

      -- Update parameters
      paramsNew = zipWith3 (\w mh vh -> w - lr * mh / (sqrt vh + eps)) params mHat vHat
      newState  = AdamState { tStep = t, mVec = mNew, vVec = vNew }
  in (newState, paramsNew)
