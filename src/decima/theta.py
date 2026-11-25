import torch
from torch.distributions import transforms


class ThetaTransformer:
    """Map physical theta = ([Fe/H], [alpha/Fe], log age, EEP) to the NPE's unbounded theta_tilde and back.

    theta is rescaled to [0, 1] over the isochrone grid's range (widened by `eps`), then logit-transformed.
    """
    param_names = ['met', 'afe', 'log_age', 'eep']

    def __init__(self, iso_grid, eps: float = 1e-6, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.eps = eps
        lo = [min(iso_grid.met), min(iso_grid.alpha), min(iso_grid.ages), iso_grid.min_eep]
        hi = [max(iso_grid.met), max(iso_grid.alpha), max(iso_grid.ages), iso_grid.max_eep]
        self.theta_min = torch.tensor([float(v) for v in lo], device=device) - eps
        self.theta_max = torch.tensor([float(v) for v in hi], device=device) + eps
        width = self.theta_max - self.theta_min
        self._transform = transforms.ComposeTransform([
            transforms.AffineTransform(loc=-self.theta_min / width, scale=1.0 / width),  # -> [0, 1]
            transforms.SigmoidTransform().inv,                                          # logit
        ])

    def inverse_transform(self, theta_tilde: torch.Tensor) -> torch.Tensor:
        """theta_tilde -> theta, i.e. (-inf, inf) -> [theta_min, theta_max]."""
        theta = self._transform.inv(theta_tilde.to(self.device))
        return theta.clamp(self.theta_min, self.theta_max)

    def forward_transform(self, theta: torch.Tensor) -> torch.Tensor:
        """theta -> theta_tilde, i.e. [theta_min, theta_max] -> (-inf, inf)."""
        return self._transform(theta.to(self.device))
