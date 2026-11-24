{-# LANGUAGE StrictData #-}
module FunctionalML.Autograd.Dual
  ( Dual (..)
  , diff
  , diffN
  ) where

-- | A Dual Number: a + b * epsilon where epsilon^2 = 0
-- 'val' represents the primal value f(x)
-- 'eps' represents the tangent / perturbation / derivative f'(x)
data Dual a = Dual
  { val :: a
  , eps :: a
  } deriving (Eq, Show)

instance (Num a) => Num (Dual a) where
  Dual u u' + Dual v v' = Dual (u + v) (u' + v')
  Dual u u' - Dual v v' = Dual (u - v) (u' - v')
  Dual u u' * Dual v v' = Dual (u * v) (u' * v + u * v')
  negate (Dual u u')     = Dual (negate u) (negate u')
  abs (Dual u u')        = Dual (abs u) (u' * signum u)
  signum (Dual u _)      = Dual (signum u) 0
  fromInteger n          = Dual (fromInteger n) 0

instance (Fractional a) => Fractional (Dual a) where
  Dual u u' / Dual v v' = Dual (u / v) ((u' * v - u * v') / (v * v))
  fromRational r        = Dual (fromRational r) 0

instance (Floating a) => Floating (Dual a) where
  pi                = Dual pi 0
  exp (Dual u u')   = let eu = exp u in Dual eu (u' * eu)
  log (Dual u u')   = Dual (log u) (u' / u)
  sin (Dual u u')   = Dual (sin u) (u' * cos u)
  cos (Dual u u')   = Dual (cos u) (- u' * sin u)
  asin (Dual u u')  = Dual (asin u) (u' / sqrt (1 - u * u))
  acos (Dual u u')  = Dual (acos u) (- u' / sqrt (1 - u * u))
  atan (Dual u u')  = Dual (atan u) (u' / (1 + u * u))
  sinh (Dual u u')  = Dual (sinh u) (u' * cosh u)
  cosh (Dual u u')  = Dual (cosh u) (u' * sinh u)
  asinh (Dual u u') = Dual (asinh u) (u' / sqrt (1 + u * u))
  acosh (Dual u u') = Dual (acosh u) (u' / sqrt (u * u - 1))
  atanh (Dual u u') = Dual (atanh u) (u' / (1 - u * u))

-- | Forward-mode automatic differentiation:
-- Computes the derivative of f at point x in a single forward pass.
diff :: (Dual Double -> Dual Double) -> Double -> Double
diff f x = eps (f (Dual x 1.0))

-- | Higher-order differentiation: compute the n-th derivative at x
diffN :: Int -> (Dual Double -> Dual Double) -> Double -> Double
diffN 0 f x = val (f (Dual x 0))
diffN 1 f x = diff f x
diffN _ _ _ = error "Higher-order forward AD beyond 1 requires nested Dual types (Dual (Dual Double))"
