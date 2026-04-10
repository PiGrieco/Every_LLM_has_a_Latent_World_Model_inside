"""
Staged training loop for the Lorentzian World Model.

Implements the three-stage protocol from Section 4.4:
  Stage 1: VAE representation learning (only for autoencoder variant)
  Stage 2: Metric + adapter + time-likeness loss
  Stage 3: + world model matching + ML + smoothness

Each stage introduces only one or two new loss terms, allowing the
geometry to stabilize before adding more complex objectives.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
from typing import Optional

from ..models.metric import MetricNetwork
from ..models.adapter import GeometryAdapter, IdentityAdapter
from ..models.lagrangian import Lagrangian
from ..models.world_model import ConditionalGaussianWorldModel
from ..training.losses import compute_total_loss, calibrate_loss_weights
from ..training.candidates import build_candidate_set_c1, build_candidate_set_c2, build_faiss_index
from ..evaluation.metrics import compute_all_metrics


class WorldModelTrainer:
    """
    Orchestrates staged training of the full pipeline:
    adapter → metric → lagrangian → world model.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # ---- Build models ----
        # Adapter: identity for synthetic, learned for real text
        if cfg.dataset in ("d0_synthetic", "d1_branching"):
            self.adapter = IdentityAdapter().to(self.device)
            input_dim = cfg.latent_dim
        else:
            self.adapter = GeometryAdapter(
                input_dim=cfg.encoder_dim,
                output_dim=cfg.latent_dim,
                hidden_dim=cfg.adapter_hidden,
                n_layers=cfg.adapter_layers,
            ).to(self.device)
            input_dim = cfg.encoder_dim

        # Metric
        self.metric = MetricNetwork(
            dim=cfg.latent_dim,
            hidden_dim=cfg.metric_hidden,
            n_layers=cfg.metric_layers,
            geometry=cfg.geometry,
        ).to(self.device)

        # Lagrangian
        use_surrogate = cfg.dataset == "d2_wikitext" and cfg.candidate_strategy != "c1"
        self.lagrangian = Lagrangian(
            metric=self.metric,
            lambda_g=cfg.lambda_g,
            lambda_sem=cfg.lambda_sem,
            use_semantic_surrogate=use_surrogate,
            latent_dim=cfg.latent_dim,
            interval_clamp_min=cfg.interval_clamp_min,
        ).to(self.device)
        self.lagrangian.set_temperature(cfg.gibbs_temperature)

        # World model
        self.world_model = ConditionalGaussianWorldModel(
            dim=cfg.latent_dim,
            hidden_dim=cfg.wm_hidden,
            n_layers=cfg.wm_layers,
        ).to(self.device)

        # ---- Optimizer ----
        # Deduplicate: lagrangian contains metric as a submodule, so
        # collecting both would register metric params twice.
        seen = {}
        for p in (
            list(self.adapter.parameters())
            + list(self.lagrangian.parameters())
            + list(self.world_model.parameters())
        ):
            seen[id(p)] = p
        self.optimizer = optim.AdamW(
            list(seen.values()), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # State
        self.current_epoch = 0
        self.knn_index = None
        self.all_states_cache = None
        self._dynamic_weights = None
        self.history = {"epoch": [], "stage": []}

    def _prepare_data(self, states, next_states, lsem=None):
        """Create a DataLoader from transition tensors."""
        tensors = [states, next_states]
        if lsem is not None:
            tensors.append(lsem)
        else:
            tensors.append(torch.zeros(len(states)))

        dataset = TensorDataset(*[t.to(self.device) for t in tensors])
        return DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True
        )

    def _cache_all_states(self, states: torch.Tensor):
        """Cache all states for kNN candidate retrieval.
        For D2: states are raw embeddings; we project through the adapter
        to get latent-space states for kNN (since energy is computed in
        latent space, neighbors must be found there too).
        """
        with torch.no_grad():
            if not isinstance(self.adapter, IdentityAdapter):
                raw = states.to(self.device)
                chunks = []
                for i in range(0, len(raw), 1024):
                    chunks.append(self.adapter(raw[i:i+1024]))
                self.all_states_cache = torch.cat(chunks, dim=0)
            else:
                self.all_states_cache = states.to(self.device)

    def _maybe_rebuild_knn(self, epoch: int):
        """Rebuild kNN index periodically for C2 candidates."""
        if (
            self.cfg.candidate_strategy in ("c1c2", "c1c2c3")
            and self.all_states_cache is not None
            and (epoch % self.cfg.knn_rebuild_every == 0 or self.knn_index is None)
        ):
            try:
                self.knn_index = build_faiss_index(self.all_states_cache)
            except ImportError:
                # FAISS not available — fall back to brute force
                self.knn_index = None

    def _build_candidates(self, s, s_next):
        """Build candidate sets based on current strategy."""
        if self.cfg.candidate_strategy == "c1":
            return build_candidate_set_c1(s, s_next, self.cfg.candidate_set_size)
        elif self.cfg.candidate_strategy in ("c1c2", "c1c2c3"):
            if self.all_states_cache is not None:
                return build_candidate_set_c2(
                    s, s_next, self.all_states_cache,
                    candidate_size=self.cfg.candidate_set_size,
                    knn_index=self.knn_index,
                )
            else:
                return build_candidate_set_c1(s, s_next, self.cfg.candidate_set_size)
        return build_candidate_set_c1(s, s_next, self.cfg.candidate_set_size)

    def train_epoch(self, loader, stage: int) -> dict:
        """
        Train for one epoch at the given stage.

        Returns dict of average losses.
        """
        self.adapter.train()
        self.metric.train()
        self.lagrangian.train()
        self.world_model.train()

        epoch_losses = {}
        n_batches = 0

        for batch in loader:
            s_raw, s_next_raw, lsem = batch

            # Apply adapter (identity for synthetic)
            s = self.adapter(s_raw) if not isinstance(self.adapter, IdentityAdapter) else s_raw
            s_next = self.adapter(s_next_raw) if not isinstance(self.adapter, IdentityAdapter) else s_next_raw

            # Build candidates
            candidates, true_idx = self._build_candidates(s.detach(), s_next.detach())

            # Compute losses
            losses = compute_total_loss(
                metric=self.metric,
                lagrangian=self.lagrangian,
                world_model=self.world_model,
                s=s,
                s_next=s_next,
                candidates=candidates,
                precomputed_lsem=lsem if lsem.abs().sum() > 0 else None,
                cfg=self.cfg,
                stage=stage,
                dynamic_weights=self._dynamic_weights,
            )

            # Backward (skip if no trainable parameter contributes to the loss,
            # which happens for euclidean geometry + identity adapter in stage 2)
            self.optimizer.zero_grad()
            if losses["total"].requires_grad:
                losses["total"].backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.metric.parameters())
                        + list(self.world_model.parameters())
                        + list(self.adapter.parameters()),
                        self.cfg.grad_clip,
                    )
                self.optimizer.step()

            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            n_batches += 1

        # Average
        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    def train(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        lsem: Optional[torch.Tensor] = None,
        eval_states: Optional[torch.Tensor] = None,
        eval_next_states: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Full staged training.

        Args:
            states: (N, D) current states (or raw embeddings for D2)
            next_states: (N, D) next states
            lsem: (N,) optional precomputed semantic costs
            eval_states/eval_next_states: held-out transitions for evaluation

        Returns:
            history: dict of lists tracking all losses and metrics per epoch
        """
        loader = self._prepare_data(states, next_states, lsem)

        # Cache states for kNN
        self._cache_all_states(states)

        total_epochs = self.cfg.stage2_epochs + self.cfg.stage3_epochs
        calibrated = False  # Track whether we've calibrated for stage 3

        print(f"\n{'='*60}")
        print(f"Training: {self.cfg.geometry} geometry | D={self.cfg.latent_dim}")
        print(f"Stage 2: {self.cfg.stage2_epochs} epochs | Stage 3: {self.cfg.stage3_epochs} epochs")
        print(f"{'='*60}\n")

        pbar = tqdm(range(total_epochs), desc="Training")

        for epoch in pbar:
            self.current_epoch = epoch

            # Determine current stage
            if epoch < self.cfg.stage2_epochs:
                stage = 2
            else:
                stage = 3

            # Auto-calibrate loss weights at the transition to stage 3
            if stage == 3 and not calibrated:
                try:
                    sample_batch = next(iter(loader))
                    s_cal = sample_batch[0]
                    sn_cal = sample_batch[1]
                    if not isinstance(self.adapter, IdentityAdapter):
                        s_cal = self.adapter(s_cal)
                        sn_cal = self.adapter(sn_cal)
                    cands, _ = self._build_candidates(s_cal.detach(), sn_cal.detach())
                    auto_weights = calibrate_loss_weights(
                        self.metric, self.lagrangian, self.world_model,
                        s_cal, sn_cal, cands, self.cfg,
                    )
                    self._dynamic_weights = auto_weights
                    tqdm.write(f"  [Auto-calibrated λ] {auto_weights}")
                except Exception:
                    pass
                calibrated = True

            # Rebuild kNN index periodically
            if stage >= 3:
                self._maybe_rebuild_knn(epoch)

            # Train one epoch
            losses = self.train_epoch(loader, stage)

            # Log
            self.history["epoch"].append(epoch)
            self.history["stage"].append(stage)
            for k, v in losses.items():
                if k not in self.history:
                    self.history[k] = []
                self.history[k].append(v)

            # Progress bar
            desc = f"S{stage}"
            for k in ["total", "time", "ml", "match"]:
                if k in losses:
                    desc += f" | {k}={losses[k]:.4f}"
            pbar.set_description(desc)

            # Periodic evaluation
            if (
                eval_states is not None
                and (epoch + 1) % self.cfg.eval_every == 0
            ):
                metrics = self.evaluate(eval_states, eval_next_states, lsem)
                for k, v in metrics.items():
                    mk = f"eval_{k}"
                    if mk not in self.history:
                        self.history[mk] = []
                    self.history[mk].append(v)
                tqdm.write(
                    f"  [Eval] M1={metrics.get('m1_timelike_rate', 0):.3f} | "
                    f"M5={metrics.get('m5_nll', 0):.3f}"
                )

        return self.history

    @torch.no_grad()
    def evaluate(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        lsem: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute M1-M6 on held-out transitions."""
        self.metric.eval()
        self.world_model.eval()

        s = states.to(self.device)
        s_next = next_states.to(self.device)

        return compute_all_metrics(
            metric=self.metric,
            world_model=self.world_model,
            lagrangian=self.lagrangian,
            s=s,
            s_next=s_next,
        )

    def save_checkpoint(self, path: str):
        """Save all model weights and training state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "adapter": self.adapter.state_dict(),
            "metric": self.metric.state_dict(),
            "lagrangian": self.lagrangian.state_dict(),
            "world_model": self.world_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "history": self.history,
            "config": self.cfg,
        }, path)

    def load_checkpoint(self, path: str):
        """Restore from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.adapter.load_state_dict(ckpt["adapter"])
        self.metric.load_state_dict(ckpt["metric"])
        self.lagrangian.load_state_dict(ckpt["lagrangian"])
        self.world_model.load_state_dict(ckpt["world_model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.current_epoch = ckpt["epoch"]
        self.history = ckpt["history"]
