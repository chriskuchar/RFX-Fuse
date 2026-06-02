"""
rfx_fuse_impute.py -- Thin re-export layer.

All imputation logic lives in rfx_impute.py. This module re-exports
the public API so that existing code using ``from rfx_fuse_impute import ...``
continues to work without changes.
"""

from rfx_impute import (
    rfx_impute,
    rfx_impute_rough,
    rfx_impute_rand,
    rfx_impute_proximity,
    rfx_impute_topk_mean,
    rfx_impute_topk_median,
    rfx_impute_knn_mean,
    rfx_impute_knn_median,
    decode_imputed,
    Imputer,
)

__all__ = [
    'rfx_impute',
    'rfx_impute_rough',
    'rfx_impute_rand',
    'rfx_impute_proximity',
    'rfx_impute_topk_mean',
    'rfx_impute_topk_median',
    'rfx_impute_knn_mean',
    'rfx_impute_knn_median',
    'decode_imputed',
    'Imputer',
]
