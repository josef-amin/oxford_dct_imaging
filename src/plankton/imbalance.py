"""Strategies for learning from imbalanced data: YOUR WORK STARTS HERE.

Detritus is ~80% of UVP6Net, and some classes have only ~100 images. A model
trained with plain cross-entropy learns to say "detritus" a lot. This module is
where the training script gets its loss function and batch sampler. Only the
plain baselines are implemented; the rest are stubs for you to fill in and
compare:

    python scripts/train_cnn.py --loss focal --sampler none
    python scripts/train_cnn.py --loss ce --sampler balanced

Each stub explains the idea and points to the original paper. These are common
starting points, not the full list: if you have a better or novel idea for
handling the imbalance, add it here (or anywhere else) and compare it.
"""

from __future__ import annotations

import numpy as np

LOSSES = ("ce", "weighted_ce", "focal", "logit_adjusted")
SAMPLERS = ("none", "balanced")


def class_counts(targets: np.ndarray, num_classes: int) -> np.ndarray:
    """Number of training examples per class index."""
    return np.bincount(np.asarray(targets), minlength=num_classes)


def make_loss(name: str, counts: np.ndarray, tau: float = 1.0):
    """Return a loss ``f(logits, targets) -> scalar`` for training.

    Args:
        name: one of LOSSES.
        counts: training examples per class (from :func:`class_counts`).
        tau: logit adjustment tuning parameter; must be > 0.
    """
    import torch  # noqa: F401 - you will need it for the TODOs below
    import torch.nn as nn

    if name == "ce":
        return nn.CrossEntropyLoss()

    if name == "weighted_ce":
        # TODO: cross-entropy where each class is weighted, e.g. by 1/count or by the
        # "effective number of samples" (Cui et al. 2019, https://arxiv.org/abs/1901.05555).
        # Hint: nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32)).
        # Remember the weights must end up on the same device as the logits.
        raise NotImplementedError("weighted_ce: see the TODO in src/plankton/imbalance.py")

    if name == "focal":
        # TODO: focal loss (Lin et al. 2017, https://arxiv.org/abs/1708.02002):
        #   FL = -(1 - p_t)^gamma * log(p_t)
        # down-weights easy, confident examples (most detritus) so training focuses on
        # hard ones. Try gamma in {1, 2, 5}.
        raise NotImplementedError("focal: see the TODO in src/plankton/imbalance.py")

    if name == "logit_adjusted":
        # logit-adjusted loss (Menon et al. 2021, https://arxiv.org/abs/2007.07314):
        # add tau * log(prior) to the logits before cross-entropy during training,
        # where prior = counts / counts.sum(). At test time use the raw logits.
        # Bonus: the same adjustment can instead be applied post hoc to a model trained
        # with plain "ce". Compare the two!

        log_prior = np.log(counts / counts.sum())   # log(prior), a measure of the class rarity in the training data.
        
        def logit_adjusted_loss(logits, targets):
            """
            Return the logit-adjusted cross-entropy loss for given logits and ground-truth targets.
            """
            prior = torch.as_tensor(
                log_prior, dtype=logits.dtype, device=logits.device # Convert log_prior into a PyTorch tensor to correctly add prior offsets
            )
            return nn.functional.cross_entropy(
                logits + tau * prior,   # Adjusted logits
                targets,                # Ground-truth class labels
            )
        
        return logit_adjusted_loss


    raise ValueError(f"loss must be one of {LOSSES}")


def make_sampler(name: str, targets: np.ndarray):
    """Return a torch Sampler for the training DataLoader, or None for plain shuffling.

    Args:
        name: one of SAMPLERS.
        targets: class index of every training example.
    """
    if name == "none":
        return None

    if name == "balanced":
        # TODO: oversample rare classes so each batch is closer to class-balanced.
        # Hint: torch.utils.data.WeightedRandomSampler with per-example weight
        # 1 / count[class], num_samples=len(targets), replacement=True.
        # Think about what this does to the number of distinct detritus images the
        # model sees per epoch, and to the predicted class frequencies at test time.
        raise NotImplementedError("balanced sampler: see the TODO in src/plankton/imbalance.py")

    raise ValueError(f"sampler must be one of {SAMPLERS}")
