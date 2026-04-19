"""Three-stage projection trainer.

Stage A: pure reconstruction.
Stage B: reconstruction + local isometry (consistency) on (h_t, h_{t+1}).
Stage C: + adversarial on-manifold (WGAN-GP). Only runs when
``cfg.use_adversarial=True``; aborts itself if the discriminator
saturates (D accuracy > 0.95 for two consecutive epochs).
"""

from __future__ import annotations

import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .autoencoder import ProjectionAutoencoder
from .config import ProjectionConfig

logger = logging.getLogger(__name__)


# ==========================================================================
# Dataset
# ==========================================================================

class PairedHiddenStateDataset(Dataset):
    """Yields ``(h_t, h_{t+1})`` pairs from per-trajectory shards.

    Filters to ``allowed_doc_ids`` so train/val splits are controlled
    by the caller at the document level. Each trajectory contributes
    at most ``len(trajectory) - 1`` pairs; pairs from different
    trajectories never mix.
    """

    def __init__(self, reader, allowed_doc_ids: set, max_pairs: Optional[int] = None):
        self._pairs: List[tuple[torch.Tensor, torch.Tensor]] = []
        for item in reader.iter_items():
            if item.get("doc_id") not in allowed_doc_ids:
                continue
            hs = item.get("hidden_states")
            if hs is None and isinstance(item.get("trajectory_a"), dict):
                hs = item["trajectory_a"].get("hidden_states")
            if hs is None or hs.shape[0] < 2:
                continue
            for t in range(hs.shape[0] - 1):
                self._pairs.append((hs[t].float().cpu(), hs[t + 1].float().cpu()))
            if max_pairs is not None and len(self._pairs) >= max_pairs:
                break

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int):
        h_t, h_tp1 = self._pairs[idx]
        return h_t, h_tp1


# ==========================================================================
# Helpers
# ==========================================================================

def _flatten_batch(h_pair: tuple) -> torch.Tensor:
    """Turn a (B, d) + (B, d) pair into (2B, d) for Stage A / per-sample ops."""
    h_t, h_tp1 = h_pair
    return torch.cat([h_t, h_tp1], dim=0)


def _gradient_penalty(
    discriminator, real: torch.Tensor, fake: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """WGAN-GP gradient penalty."""
    alpha = torch.rand(real.shape[0], 1, device=real.device)
    interpolated = alpha * real + (1.0 - alpha) * fake
    interpolated.requires_grad_(True)
    d_interp = discriminator(interpolated)
    grads = torch.autograd.grad(
        outputs=d_interp, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gp = ((grads.norm(2, dim=-1) - 1.0) ** 2).mean()
    return weight * gp


# ==========================================================================
# Trainer
# ==========================================================================

class ProjectionTrainer:
    """Runs the three stages end-to-end and saves per-stage checkpoints."""

    def __init__(
        self,
        cfg: ProjectionConfig,
        autoencoder: ProjectionAutoencoder,
        d_model: int,
    ) -> None:
        self.cfg = cfg
        self.ae = autoencoder
        self.d_model = int(d_model)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.ae.to(self.device)
        self.history: dict = {"stage_a": [], "stage_b": [], "stage_c": []}

    # ---- Stage A ----
    def train_stage_a(self, loader: Iterable) -> dict:
        opt = torch.optim.Adam(
            list(self.ae.encoder.parameters()) + list(self.ae.decoder.parameters()),
            lr=self.cfg.stage_a_lr,
        )
        best = float("inf")
        no_improve = 0
        for epoch in range(self.cfg.stage_a_epochs):
            self.ae.train()
            losses: List[float] = []
            for h_t, h_tp1 in loader:
                h = torch.cat([h_t, h_tp1], dim=0).to(self.device)
                loss = self.ae.reconstruction_loss(h)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            mean = sum(losses) / max(1, len(losses))
            self.history["stage_a"].append({"epoch": epoch, "loss": mean})
            logger.info("[stage A] epoch %d mse=%.4f", epoch, mean)
            if mean < best - 1e-6:
                best = mean
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 3:
                    logger.info("[stage A] early stop at epoch %d", epoch)
                    break
        return {"best_mse": best, "epochs_run": len(self.history["stage_a"])}

    # ---- Stage B ----
    def train_stage_b(self, loader: Iterable) -> dict:
        opt = torch.optim.Adam(
            list(self.ae.encoder.parameters())
            + list(self.ae.decoder.parameters())
            + [self.ae.log_alpha],
            lr=self.cfg.stage_b_lr,
        )
        best = float("inf")
        for epoch in range(self.cfg.stage_b_epochs):
            self.ae.train()
            losses: List[float] = []
            for h_t, h_tp1 in loader:
                h_t = h_t.to(self.device)
                h_tp1 = h_tp1.to(self.device)
                # Reconstruction on the union of both halves.
                h = torch.cat([h_t, h_tp1], dim=0)
                l_rec = self.ae.reconstruction_loss(h)
                l_cons = self.ae.consistency_loss(h_t, h_tp1)
                loss = l_rec + self.cfg.lambda_consistency * l_cons
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            mean = sum(losses) / max(1, len(losses))
            self.history["stage_b"].append({"epoch": epoch, "loss": mean})
            logger.info("[stage B] epoch %d loss=%.4f α=%.3f",
                        epoch, mean, float(self.ae.alpha))
            if mean < best - 1e-6:
                best = mean
        return {"best_loss": best, "epochs_run": len(self.history["stage_b"])}

    # ---- Stage C ----
    def train_stage_c(self, loader: Iterable) -> dict:
        """Adversarial on-manifold training (only if discriminator exists)."""
        if self.ae.discriminator is None:
            logger.info("[stage C] skipped (use_adversarial=False)")
            return {"skipped": True}

        D = self.ae.discriminator
        opt_g = torch.optim.Adam(
            list(self.ae.encoder.parameters())
            + list(self.ae.decoder.parameters())
            + [self.ae.log_alpha],
            lr=self.cfg.stage_c_lr_g,
        )
        opt_d = torch.optim.Adam(D.parameters(), lr=self.cfg.stage_c_lr_d)

        prev_d_acc = 0.0
        saturation_streak = 0
        for epoch in range(self.cfg.stage_c_epochs):
            self.ae.train()
            d_losses: List[float] = []
            g_losses: List[float] = []
            d_accs: List[float] = []
            for h_t, h_tp1 in loader:
                h = torch.cat([h_t, h_tp1], dim=0).to(self.device)

                # ---- Discriminator step(s) ----
                for _ in range(self.cfg.discriminator_steps_per_g):
                    with torch.no_grad():
                        _, h_hat = self.ae(h)
                    real_score = D(h)
                    fake_score = D(h_hat.detach())
                    wgan = fake_score.mean() - real_score.mean()
                    gp = _gradient_penalty(
                        D, h, h_hat.detach(), self.cfg.discriminator_gp_weight,
                    )
                    d_loss = wgan + gp
                    opt_d.zero_grad()
                    d_loss.backward()
                    opt_d.step()
                    d_losses.append(float(d_loss.item()))
                    # Coarse accuracy: real>0 and fake<0.
                    with torch.no_grad():
                        acc = 0.5 * ((real_score > 0).float().mean().item()
                                     + (fake_score < 0).float().mean().item())
                        d_accs.append(acc)

                # ---- Generator step ----
                _, h_hat = self.ae(h)
                l_rec = ((h - h_hat) ** 2).mean()
                l_cons = self.ae.consistency_loss(
                    h_t.to(self.device), h_tp1.to(self.device),
                )
                l_adv = -D(h_hat).mean()
                loss = (
                    l_rec
                    + self.cfg.lambda_consistency * l_cons
                    + self.cfg.lambda_adversarial * l_adv
                )
                opt_g.zero_grad()
                loss.backward()
                opt_g.step()
                g_losses.append(float(loss.item()))

            mean_d = sum(d_losses) / max(1, len(d_losses))
            mean_g = sum(g_losses) / max(1, len(g_losses))
            mean_acc = sum(d_accs) / max(1, len(d_accs))
            self.history["stage_c"].append({
                "epoch": epoch, "d_loss": mean_d, "g_loss": mean_g,
                "d_acc": mean_acc,
            })
            logger.info(
                "[stage C] epoch %d d=%.4f g=%.4f d_acc=%.3f",
                epoch, mean_d, mean_g, mean_acc,
            )
            if mean_acc > 0.95:
                saturation_streak += 1
            else:
                saturation_streak = 0
            if saturation_streak >= 2:
                logger.warning(
                    "[stage C] discriminator saturated (d_acc > 0.95 for 2 "
                    "consecutive epochs); aborting stage C."
                )
                return {
                    "aborted": True, "reason": "D_saturation",
                    "epochs_run": len(self.history["stage_c"]),
                }
            prev_d_acc = mean_acc
        return {"aborted": False, "epochs_run": len(self.history["stage_c"])}

    # ---- Orchestrator ----
    def train(
        self, loader: Iterable, stages: str = "abc",
        save_dir: Optional[str] = None,
    ) -> dict:
        results = {"stages_run": stages, "history": self.history}
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        if "a" in stages:
            results["stage_a"] = self.train_stage_a(loader)
            if save_dir:
                self.save(os.path.join(save_dir, "ckpt_stage_a.pt"))
        if "b" in stages:
            results["stage_b"] = self.train_stage_b(loader)
            if save_dir:
                self.save(os.path.join(save_dir, "ckpt_stage_b.pt"))
        if "c" in stages:
            if self.cfg.use_adversarial:
                results["stage_c"] = self.train_stage_c(loader)
                if save_dir:
                    self.save(os.path.join(save_dir, "ckpt_stage_c.pt"))
            else:
                results["stage_c"] = {"skipped": True, "reason": "use_adversarial=False"}
        if save_dir:
            self.save(os.path.join(save_dir, "final.pt"))
        return results

    # ---- I/O ----
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.ae.state_dict(),
            "cfg": self.cfg.__dict__,
            "d_model": self.d_model,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.ae.load_state_dict(ckpt["state_dict"])
