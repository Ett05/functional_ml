{-# LANGUAGE StrictData #-}
module FunctionalML.NN.Loss
  ( mseLoss
  , mseLossTape
  , bceLoss
  , bceLossTape
  ) where

import Control.Monad (zipWithM, foldM)
import FunctionalML.Autograd.Reverse

-- | Standard Mean Squared Error (MSE) loss
mseLoss :: [Double] -> [Double] -> Double
mseLoss preds targets =
  let sqErrors = zipWith (\p y -> (p - y) ** 2) preds targets
  in sum sqErrors / fromIntegral (length preds)

-- | Differentiable MSE loss inside the TapeM monad
mseLossTape :: [Expr] -> [Double] -> TapeM Expr
mseLossTape preds targets = do
  sqErrors <- zipWithM (\p y -> do
                          yE <- constExpr y
                          diffE <- subE p yE
                          mulE diffE diffE
                       ) preds targets
  totalErr <- foldM addE (head sqErrors) (tail sqErrors)
  nConst <- constExpr (fromIntegral (length targets))
  divE totalErr nConst

-- | Binary Cross Entropy (BCE) loss
bceLoss :: [Double] -> [Double] -> Double
bceLoss preds targets =
  let eps = 1e-7
      losses = zipWith (\p y -> - (y * log (max eps p) + (1 - y) * log (max eps (1 - p)))) preds targets
  in sum losses / fromIntegral (length preds)

-- | Differentiable BCE loss inside the TapeM monad
bceLossTape :: [Expr] -> [Double] -> TapeM Expr
bceLossTape preds targets = do
  losses <- zipWithM (\p y -> do
                        yE <- constExpr y
                        oneE <- constExpr 1.0
                        logP <- logE p
                        term1 <- mulE yE logP
                        oneMinusY <- constExpr (1.0 - y)
                        oneMinusP <- subE oneE p
                        logOneMinusP <- logE oneMinusP
                        term2 <- mulE oneMinusY logOneMinusP
                        sumTerms <- addE term1 term2
                        negE sumTerms
                     ) preds targets
  totalLoss <- foldM addE (head losses) (tail losses)
  nConst <- constExpr (fromIntegral (length targets))
  divE totalLoss nConst
