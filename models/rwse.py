"""Random-walk structural encoding (RWSE) for molecular graphs.

For each atom `i` the encoding is the probability that a random walk starting
at `i` is back at `i` after exactly `k` steps, for k = 1..K:

    RWSE_i = [ (D^-1 A)^1_ii, (D^-1 A)^2_ii, ..., (D^-1 A)^K_ii ]

Why this and not more parameters: message passing on a molecular graph is
bounded by 1-WL, which cannot count cycles. Two atoms with identical
neighbourhoods out to depth `T` get identical embeddings even when one sits in
a fused bicyclic and the other in a chain -- a distinction that matters for
metabolic clearance. Return probability is exactly the quantity that separates
them, because a walk can only come back to where it started by going around a
cycle: `RWSE[k]` is non-zero at k equal to a ring size the atom belongs to (and
at multiples of it). The encoding is a fixed function of the graph, so it adds
structural information the network cannot derive on its own, at zero parameter
cost in the encoder.

Values are probabilities in [0, 1] and need no rescaling before use.
"""

import numpy as np
from rdkit import Chem


def compute_rwse(mol: Chem.Mol, k: int) -> np.ndarray | None:
    """Return the `(n_atoms, k)` RWSE matrix for `mol`, or None when `k <= 0`.

    Returning None at `k = 0` is what lets the caller skip the extra-feature
    path entirely, so `rwse_k = 0` reproduces the plain featurizer exactly
    rather than concatenating a zero-width array.
    """
    if k <= 0:
        return None

    n = mol.GetNumAtoms()
    # chemprop's featurizer emits a single zero row for an empty molecule; match
    # that shape so the horizontal concatenation downstream still lines up.
    if n == 0:
        return np.zeros((1, k), dtype=np.single)

    A = np.zeros((n, n), dtype=np.float64)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[i, j] = A[j, i] = 1.0

    deg = A.sum(axis=1)
    # A disconnected atom has no walk to return on. `where=` leaves its inverse
    # degree at 0, which zeroes its row and gives it an all-zero encoding --
    # correct, and it keeps the division from emitting a warning.
    inv_deg = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    P = inv_deg[:, None] * A  # row-stochastic transition matrix D^-1 A

    out = np.empty((n, k), dtype=np.single)
    walk = np.eye(n, dtype=np.float64)
    for step in range(k):
        walk = walk @ P
        out[:, step] = np.diagonal(walk)

    return out


def rwse_dims(rwse_k: int, rwse_at: str) -> tuple[int, int]:
    """Split `rwse_k` into (extra atom-feature width, atom-descriptor width).

    The two modes enter the network at different points, and each one sizes a
    different weight matrix, so exactly one of these is non-zero:

    * "input"   -> `V_f`, concatenated onto the atom features before message
      passing. This widens `d_v`, which sizes both `W_i` and `W_o`, so the
      encoding conditions the message passing itself.
    * "readout" -> `V_d`, which is only seen by `finalize` via `W_d`. Message
      passing stays purely chemical and the structure is mixed in afterwards.
    """
    if rwse_k <= 0:
        return 0, 0
    if rwse_at == "input":
        return rwse_k, 0
    if rwse_at == "readout":
        return 0, rwse_k
    raise ValueError(f"unknown rwse_at: {rwse_at!r}")
