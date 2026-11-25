"""MIST isochrones on a regular grid in theta = ([Fe/H], [alpha/Fe], log(age/yr), EEP), linearly interpolated with torch.

Loaded from the `.npz` written by isosbi's `export_decima_models.py` (axes, the value cube and its column names). The
interpolation reproduces scipy's linear `RegularGridInterpolator` exactly: the same cell search (grid[i] <= x < grid[i+1],
end cells for points on the upper edge), the same 16-corner weights summed in the same order, and NaN for points outside
the grid or with a NaN coordinate. A NaN at any of the 16 corners makes the result NaN (0 * NaN = NaN), which marks
off-isochrone draws as invalid downstream.

Columns: the bands G, BP, RP (Gaia EDR3) and J, H, Ks (2MASS), log_Teff, log_g and ln_dm_dEEP_prior (log IMF prior per
unit EEP).
"""
import itertools
from typing import Sequence

import numpy as np
import torch


class IsochroneGrid:
    _LOOKUP_CHUNK = 500_000  # query points interpolated at once (bounds memory)

    def __init__(self, met, alpha, ages, eep_grid, min_eep, max_eep, grid_cube, columns, device="cpu"):
        self.met, self.alpha, self.ages, self.eep_grid = met, alpha, ages, eep_grid
        self.min_eep, self.max_eep = float(min_eep), float(max_eep)
        self.grid_cube = grid_cube  # (n_met, n_alpha, n_age, n_eep, n_cols) float64, NaN off the isochrones
        self.col_map = list(columns)
        self._col_map = {c: i for i, c in enumerate(self.col_map)}
        self.to(device)

    @classmethod
    def load(cls, path, device="cpu") -> "IsochroneGrid":
        with np.load(path) as d:
            return cls(*(d[k] for k in ("met", "alpha", "ages", "eep_grid", "min_eep", "max_eep", "grid_cube")),
                       columns=[str(c) for c in d["columns"]], device=device)

    def to(self, device) -> "IsochroneGrid":
        """Move the interpolation tensors to `device` (the numpy grid_cube stays on the CPU)."""
        self.device = torch.device(device)
        self._axes_t = tuple(
            torch.as_tensor(a, dtype=torch.float64, device=self.device)
            for a in (self.met, self.alpha, self.ages, self.eep_grid)
        )
        cube = torch.from_numpy(np.ascontiguousarray(self.grid_cube, dtype=np.float64))
        # (n_nodes, n_cols): node (i0, i1, i2, i3) is row ((i0 * n1 + i1) * n2 + i2) * n3 + i3
        self._cube_t = cube.reshape(-1, cube.shape[-1]).to(self.device)
        shape = self.grid_cube.shape[:4]
        self._strides = tuple(int(np.prod(shape[d + 1:])) for d in range(4))
        return self

    def _check_cols(self, cols: Sequence[str]) -> tuple:
        cols = tuple(cols)
        missing = [c for c in cols if c not in self._col_map]
        if missing:
            raise ValueError(f"Column(s) {missing} not found. Available: {self.col_map}")
        return cols

    def _interpolate(self, x: torch.Tensor, col_idx: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of the columns `col_idx` at `x` (n, 4) float64; returns (n, len(col_idx))."""
        lower, frac = [], []
        for d, grid in enumerate(self._axes_t):
            xd = x[:, d].contiguous()
            i = (torch.searchsorted(grid, xd, right=True) - 1).clamp(0, len(grid) - 2)
            lower.append(i)
            frac.append((xd - grid[i]) / (grid[i + 1] - grid[i]))

        # scipy's _evaluate_linear: corners in itertools.product order (lower before upper, first axis
        # slowest), weight = 1 * w0 * w1 * w2 * w3, value = 0 + term_0 + term_1 + ... (bit-identical sums)
        value = torch.zeros(len(x), len(col_idx), dtype=torch.float64, device=x.device)
        for corner in itertools.product((0, 1), repeat=4):
            node = torch.zeros_like(lower[0])
            weight = torch.ones_like(frac[0])
            for d, upper in enumerate(corner):
                node = node + (lower[d] + upper) * self._strides[d]
                weight = weight * (frac[d] if upper else 1 - frac[d])
            value = value + self._cube_t[node[:, None], col_idx[None, :]] * weight[:, None]

        lo = torch.stack([g[0] for g in self._axes_t])
        hi = torch.stack([g[-1] for g in self._axes_t])
        bad = (x < lo).any(dim=1) | (x > hi).any(dim=1) | torch.isnan(x).any(dim=1)
        return value.masked_fill(bad[:, None], float("nan"))

    def lookup(self, pts, cols: Sequence[str]):
        """
        Interpolate several columns at once.

        `pts` is (n_points, 4), numpy or torch. Returns (len(cols), n_points): a numpy array for
        numpy input, a float64 tensor on the grid's device for torch input.
        """
        cols = self._check_cols(cols)
        as_tensor = torch.is_tensor(pts)
        x = (pts if as_tensor else torch.from_numpy(np.asarray(pts, dtype=float))).to(self.device, torch.float64)
        col_idx = torch.tensor([self._col_map[c] for c in cols], device=self.device)
        out = torch.empty(len(cols), len(x), dtype=torch.float64, device=self.device)
        for lo in range(0, len(x), self._LOOKUP_CHUNK):
            hi = min(lo + self._LOOKUP_CHUNK, len(x))
            out[:, lo:hi] = self._interpolate(x[lo:hi], col_idx).T
        return out if as_tensor else out.cpu().numpy()

    def get_columns(self, pts, cols: Sequence[str]) -> dict:
        """Return {col: (n_points,) values} for several columns in a single pass."""
        cols = self._check_cols(cols)
        out = self.lookup(pts, cols)
        return {c: out[i] for i, c in enumerate(cols)}
