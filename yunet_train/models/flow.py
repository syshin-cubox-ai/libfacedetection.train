from __future__ import annotations

import torch
import torch.nn as nn


class RealNVP(nn.Module):
    """Flow-based generative model used by the YOLO26 RLE keypoint loss.

    Learns the distribution of normalized keypoint regression errors so the
    RLE loss can score predictions by log-likelihood.

    References:
        https://arxiv.org/abs/1605.08803
        https://arxiv.org/abs/2107.11291
    """

    @staticmethod
    def nets() -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    @staticmethod
    def nett() -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2)
        )

    @property
    def prior(self) -> torch.distributions.MultivariateNormal:
        return torch.distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super().__init__()
        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer(
            "mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32)
        )

        self.s = nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map data space to latent space, returning the log-determinant of the Jacobian."""
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det
