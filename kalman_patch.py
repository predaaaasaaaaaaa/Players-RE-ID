"""
Patch BoxMOT's Kalman filter to handle non-positive-definite matrices.
Import this BEFORE creating the tracker.
"""
import numpy as np
import scipy.linalg

# Save original
_original_cho_factor = scipy.linalg.cho_factor


def _safe_cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """Fallback: force positive definite if Cholesky fails."""
    try:
        return _original_cho_factor(a, lower=lower, overwrite_a=overwrite_a,
                                     check_finite=check_finite)
    except np.linalg.LinAlgError:
        # Force positive definite — keep retrying with increasing regularization
        for eps in [1e-4, 1e-3, 1e-2, 1e-1]:
            try:
                a_fixed = a.copy() + np.eye(a.shape[0]) * eps
                return _original_cho_factor(a_fixed, lower=lower, overwrite_a=False,
                                             check_finite=check_finite)
            except np.linalg.LinAlgError:
                continue
        # Last resort: return identity-like result
        n = a.shape[0]
        return (np.eye(n), lower)

# Apply patch
scipy.linalg.cho_factor = _safe_cho_factor
print("[Patch] Kalman filter cho_factor patched for numerical stability")