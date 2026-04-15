"""
Patch BoxMOT's Kalman filter to handle non-positive-definite matrices.
Import this BEFORE creating the tracker.
"""
import numpy as np
import scipy.linalg

# Save original
_original_cho_factor = scipy.linalg.cho_factor


def _safe_cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """Fallback to LU decomposition if Cholesky fails."""
    try:
        return _original_cho_factor(a, lower=lower, overwrite_a=overwrite_a,
                                     check_finite=check_finite)
    except np.linalg.LinAlgError:
        # Force positive definite via eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(a)
        eigvals = np.maximum(eigvals, 1e-4)
        a_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return _original_cho_factor(a_fixed, lower=lower, overwrite_a=False,
                                     check_finite=check_finite)


# Apply patch
scipy.linalg.cho_factor = _safe_cho_factor
print("[Patch] Kalman filter cho_factor patched for numerical stability")