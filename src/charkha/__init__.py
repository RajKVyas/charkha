"""CHARKHA — depth-recurrent hybrid language model for consumer GPUs."""

from .config import CharkhaConfig
from ._model import Charkha
from ._modules import have_fla
from ._optim import (
    build_optimizers,
    build_symmetry_optimizers,
    wsd_lr_mult,
    Muon,
    NormM,
    install_grad_release,
    clip_grads_mixed,
)

__all__ = [
    "CharkhaConfig",
    "Charkha",
    "build_optimizers",
    "build_symmetry_optimizers",
    "wsd_lr_mult",
    "Muon",
    "NormM",
    "install_grad_release",
    "clip_grads_mixed",
    "have_fla",
]
