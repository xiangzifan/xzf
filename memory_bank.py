"""Reliability-aware evolving paired support bank for IN2R.

The bank deliberately contains no trainable parameters.  It stores detached
image/text embeddings and metadata supplied by the peer-network data split.
Three modes are provided from the same implementation so that experiments can
isolate the contribution of each mechanism:

``fifo``
    Circular FIFO with the same retrieval rule as A5-Delete.
``id_refresh``
    One slot per stable training record, reliability-aware feature refresh,
    and exact self-record exclusion during retrieval.
``quality_diverse``
    ``id_refresh`` plus reliability/freshness reranking and MMR support-set
    selection in the paired value modality.

Synthetic clean/noisy ground-truth labels are never accepted by this module.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.nn.functional as F


_VALID_MODES = {"fifo", "id_refresh", "quality_diverse"}


class MemoryQueue:
    """Paired feature bank with controlled A5-compatible ablations."""

    def __init__(
        self,
        queue_size: int = 2048,
        dim: int = 1024,
        mode: str = "quality_diverse",
        candidate_pool_size: int = 20,
        quality_weight: float = 0.25,
        freshness_weight: float = 0.10,
        freshness_tau: float = 2.0,
        mmr_balance: float = 0.80,
        alpha_min: float = 0.20,
        alpha_max: float = 0.80,
        quality_momentum: float = 0.50,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
    ) -> None:
        if queue_size < 1 or dim < 1:
            raise ValueError("queue_size and dim must be positive")
        if mode not in _VALID_MODES:
            raise ValueError("Unsupported memory mode: {}".format(mode))
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be positive")
        if quality_weight < 0 or freshness_weight < 0:
            raise ValueError("memory reranking weights must be nonnegative")
        if freshness_tau <= 0:
            raise ValueError("freshness_tau must be positive")
        if not 0 <= mmr_balance <= 1:
            raise ValueError("mmr_balance must be in [0, 1]")
        if not 0 <= alpha_min <= alpha_max <= 1:
            raise ValueError("require 0 <= alpha_min <= alpha_max <= 1")
        if not 0 <= quality_momentum <= 1:
            raise ValueError("quality_momentum must be in [0, 1]")
        if eps <= 0:
            raise ValueError("eps must be positive")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.queue_size = int(queue_size)
        self.dim = int(dim)
        self.mode = mode
        self.candidate_pool_size = int(candidate_pool_size)
        self.quality_weight = float(quality_weight)
        self.freshness_weight = float(freshness_weight)
        self.freshness_tau = float(freshness_tau)
        self.mmr_balance = float(mmr_balance)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.quality_momentum = float(quality_momentum)
        self.eps = float(eps)

        self.img_queue = torch.zeros(self.queue_size, self.dim, device=device)
        self.txt_queue = torch.zeros(self.queue_size, self.dim, device=device)
        self.record_ids = torch.full(
            (self.queue_size,), -1, dtype=torch.long, device=device
        )
        self.quality = torch.zeros(self.queue_size, device=device)
        self.last_seen = torch.full(
            (self.queue_size,), -1, dtype=torch.long, device=device
        )
        self.update_count = torch.zeros(
            self.queue_size, dtype=torch.long, device=device
        )

        self.ptr = 0
        self.size = 0
        self.is_full = False
        self._id_to_slot: Dict[int, int] = {}
        self._stats = defaultdict(float)

    @property
    def device(self) -> torch.device:
        return self.img_queue.device

    @property
    def current_size(self) -> int:
        return self.size

    def _validate_feature_batch(
        self, img_feats: torch.Tensor, txt_feats: torch.Tensor
    ) -> None:
        if img_feats.ndim != 2 or txt_feats.ndim != 2:
            raise ValueError("memory features must have shape [batch, dim]")
        if img_feats.shape != txt_feats.shape:
            raise ValueError("paired image/text features must have identical shapes")
        if img_feats.size(1) != self.dim:
            raise ValueError("feature dimension does not match memory dimension")
        if img_feats.device != self.device or txt_feats.device != self.device:
            raise ValueError("memory inputs and storage must be on the same device")
        if not torch.isfinite(img_feats).all() or not torch.isfinite(txt_feats).all():
            raise ValueError("memory features must be finite")

    def _as_metadata(
        self,
        record_ids,
        clean_prob,
        batch_size: int,
    ):
        ids = torch.as_tensor(record_ids, dtype=torch.long, device=self.device).reshape(-1)
        probs = torch.as_tensor(
            clean_prob, dtype=self.quality.dtype, device=self.device
        ).reshape(-1)
        if ids.numel() != batch_size or probs.numel() != batch_size:
            raise ValueError("record_ids and clean_prob must align with feature rows")
        if (ids < 0).any():
            raise ValueError("record_ids must be nonnegative")
        if not torch.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
            raise ValueError("clean probabilities must be finite and in [0, 1]")
        return ids, probs

    @torch.no_grad()
    def enqueue(
        self,
        img_feats: torch.Tensor,
        txt_feats: torch.Tensor,
        record_ids=None,
        clean_prob=None,
        current_epoch: int = 0,
    ) -> None:
        """Insert or refresh paired features.

        ``fifo`` ignores metadata and reproduces circular queue behaviour.
        ID-aware modes require stable record IDs and peer clean probabilities.
        When a full ID-aware bank receives a previously unseen record, the next
        FIFO slot is replaced; semantic eviction is intentionally deferred to a
        separately controlled compact-memory experiment.
        """
        self._validate_feature_batch(img_feats, txt_feats)
        batch_size = img_feats.size(0)
        if batch_size == 0:
            return
        if batch_size > self.queue_size:
            raise ValueError("queue capacity must be at least the enqueue batch size")
        if current_epoch < 0:
            raise ValueError("current_epoch must be nonnegative")

        img_feats = img_feats.detach()
        txt_feats = txt_feats.detach()
        self._stats["enqueue_rows"] += batch_size

        if self.mode == "fifo":
            slots = (
                torch.arange(batch_size, device=self.device, dtype=torch.long)
                + self.ptr
            ) % self.queue_size
            self.img_queue[slots] = img_feats
            self.txt_queue[slots] = txt_feats
            self.ptr = (self.ptr + batch_size) % self.queue_size
            self.size = min(self.queue_size, self.size + batch_size)
            self.is_full = self.size == self.queue_size
            self._stats["inserted_rows"] += batch_size
            return

        ids, probs = self._as_metadata(record_ids, clean_prob, batch_size)
        for row in range(batch_size):
            record_id = int(ids[row].item())
            probability = probs[row]
            slot = self._id_to_slot.get(record_id)
            if slot is not None:
                alpha = self.alpha_min + (
                    self.alpha_max - self.alpha_min
                ) * float(probability.item())
                self.img_queue[slot] = F.normalize(
                    (1.0 - alpha) * self.img_queue[slot]
                    + alpha * img_feats[row],
                    p=2,
                    dim=0,
                    eps=self.eps,
                )
                self.txt_queue[slot] = F.normalize(
                    (1.0 - alpha) * self.txt_queue[slot]
                    + alpha * txt_feats[row],
                    p=2,
                    dim=0,
                    eps=self.eps,
                )
                self.quality[slot] = (
                    self.quality_momentum * self.quality[slot]
                    + (1.0 - self.quality_momentum) * probability
                )
                self.last_seen[slot] = int(current_epoch)
                self.update_count[slot] += 1
                self._stats["updated_rows"] += 1
                continue

            if self.size < self.queue_size:
                slot = self.size
                self.size += 1
                self.ptr = self.size % self.queue_size
            else:
                slot = self.ptr
                previous_id = int(self.record_ids[slot].item())
                if previous_id >= 0:
                    self._id_to_slot.pop(previous_id, None)
                self.ptr = (self.ptr + 1) % self.queue_size
                self._stats["evicted_rows"] += 1

            self.img_queue[slot] = F.normalize(
                img_feats[row], p=2, dim=0, eps=self.eps
            )
            self.txt_queue[slot] = F.normalize(
                txt_feats[row], p=2, dim=0, eps=self.eps
            )
            self.record_ids[slot] = record_id
            self.quality[slot] = probability
            self.last_seen[slot] = int(current_epoch)
            self.update_count[slot] = 1
            self._id_to_slot[record_id] = slot
            self._stats["inserted_rows"] += 1

        self.is_full = self.size == self.queue_size

    def _standardize(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=1, keepdim=True)
        variance = ((values - mean) ** 2).mean(dim=1, keepdim=True)
        return (values - mean) / variance.sqrt().clamp_min(self.eps)

    @torch.no_grad()
    def _select_supports(
        self,
        query_feats: torch.Tensor,
        key_bank: torch.Tensor,
        value_bank: torch.Tensor,
        k: int,
        query_ids=None,
        current_epoch: int = 0,
    ):
        if query_feats.ndim != 2 or query_feats.size(1) != self.dim:
            raise ValueError("query features must have shape [batch, dim]")
        if query_feats.device != self.device:
            raise ValueError("query features and memory must be on the same device")
        if k < 1:
            raise ValueError("k must be positive")
        if self.size < k:
            self._stats["insufficient_support_calls"] += 1
            return None
        if current_epoch < 0:
            raise ValueError("current_epoch must be nonnegative")

        valid_keys = key_bank[: self.size]
        valid_values = value_bank[: self.size]
        similarities = torch.mm(query_feats.detach(), valid_keys.t())
        available = torch.ones_like(similarities, dtype=torch.bool)

        if self.mode != "fifo" and query_ids is not None:
            ids = torch.as_tensor(
                query_ids, dtype=torch.long, device=self.device
            ).reshape(-1)
            if ids.numel() != query_feats.size(0):
                raise ValueError("query_ids must align with query rows")
            same_record = ids[:, None] == self.record_ids[: self.size][None, :]
            available &= ~same_record
            similarities = similarities.masked_fill(same_record, -torch.inf)
            self._stats["self_matches_masked"] += int(same_record.sum().item())

        minimum_available = int(available.sum(dim=1).min().item())
        if minimum_available < k:
            self._stats["insufficient_support_calls"] += 1
            return None

        candidate_count = k
        if self.mode == "quality_diverse":
            candidate_count = min(
                max(k, self.candidate_pool_size), minimum_available
            )

        candidate_scores, candidate_indices = torch.topk(
            similarities, candidate_count, dim=1
        )
        candidate_values = valid_values[candidate_indices]

        if self.mode != "quality_diverse":
            selected_positions = torch.arange(k, device=self.device)[None, :].expand(
                query_feats.size(0), -1
            )
        else:
            candidate_quality = self.quality[: self.size][candidate_indices]
            candidate_age = (
                int(current_epoch) - self.last_seen[: self.size][candidate_indices]
            ).clamp_min(0).to(candidate_scores.dtype)
            candidate_freshness = torch.exp(-candidate_age / self.freshness_tau)
            base_score = (
                self._standardize(candidate_scores)
                + self.quality_weight * self._standardize(candidate_quality)
                + self.freshness_weight * self._standardize(candidate_freshness)
            )

            normalized_values = F.normalize(
                candidate_values, p=2, dim=-1, eps=self.eps
            )
            selected = []
            selected_mask = torch.zeros_like(base_score, dtype=torch.bool)
            batch_positions = torch.arange(query_feats.size(0), device=self.device)
            for support_index in range(k):
                if support_index == 0:
                    selection_score = base_score
                else:
                    chosen_values = torch.stack(
                        [
                            normalized_values[batch_positions, position]
                            for position in selected
                        ],
                        dim=1,
                    )
                    redundancy = torch.matmul(
                        normalized_values, chosen_values.transpose(1, 2)
                    ).max(dim=-1).values
                    selection_score = (
                        self.mmr_balance * base_score
                        - (1.0 - self.mmr_balance) * redundancy
                    )
                selection_score = selection_score.masked_fill(
                    selected_mask, -torch.inf
                )
                position = selection_score.argmax(dim=1)
                selected.append(position)
                selected_mask[batch_positions, position] = True
            selected_positions = torch.stack(selected, dim=1)

            selected_quality = candidate_quality.gather(1, selected_positions)
            selected_freshness = candidate_freshness.gather(1, selected_positions)
            self._stats["selected_quality_sum"] += float(
                selected_quality.sum().item()
            )
            self._stats["selected_freshness_sum"] += float(
                selected_freshness.sum().item()
            )
            self._stats["selected_metadata_count"] += selected_quality.numel()

            raw_topk = torch.arange(k, device=self.device)[None, None, :]
            overlap = (selected_positions[:, :, None] == raw_topk).any(dim=-1)
            self._stats["raw_topk_overlap_sum"] += float(overlap.sum().item())
            self._stats["raw_topk_overlap_count"] += overlap.numel()

        selected_indices = candidate_indices.gather(1, selected_positions)
        selected_values = valid_values[selected_indices]

        if k > 1:
            normalized_selected = F.normalize(
                selected_values, p=2, dim=-1, eps=self.eps
            )
            pairwise = torch.matmul(
                normalized_selected, normalized_selected.transpose(1, 2)
            )
            upper = torch.triu(
                torch.ones(k, k, dtype=torch.bool, device=self.device), diagonal=1
            )
            selected_pairs = pairwise[:, upper]
            self._stats["selected_pair_similarity_sum"] += float(
                selected_pairs.sum().item()
            )
            self._stats["selected_pair_similarity_count"] += selected_pairs.numel()

        self._stats["retrieval_calls"] += 1
        self._stats["retrieval_queries"] += query_feats.size(0)
        return selected_values, selected_indices

    def _retrieve(
        self,
        query_feats: torch.Tensor,
        key_bank: torch.Tensor,
        value_bank: torch.Tensor,
        k: int,
        refiner_node=None,
        query_ids=None,
        current_epoch: int = 0,
    ):
        selected = self._select_supports(
            query_feats,
            key_bank,
            value_bank,
            k,
            query_ids=query_ids,
            current_epoch=current_epoch,
        )
        if selected is None:
            return None
        neighbor_values, _ = selected
        if refiner_node is not None:
            return refiner_node(neighbor_values)
        return F.normalize(
            neighbor_values.mean(dim=1), p=2, dim=-1, eps=self.eps
        )

    def retrieve_nearest_text(
        self,
        query_img_feats: torch.Tensor,
        k: int = 1,
        refiner_node=None,
        img_refiner_node=None,
        query_ids=None,
        current_epoch: int = 0,
    ):
        del img_refiner_node  # Kept only for compatibility with the A5 call site.
        return self._retrieve(
            query_img_feats,
            self.img_queue,
            self.txt_queue,
            k,
            refiner_node=refiner_node,
            query_ids=query_ids,
            current_epoch=current_epoch,
        )

    def retrieve_nearest_img(
        self,
        query_text_feats: torch.Tensor,
        k: int = 1,
        refiner_node=None,
        txt_refiner_node=None,
        query_ids=None,
        current_epoch: int = 0,
    ):
        del txt_refiner_node  # Kept only for compatibility with the A5 call site.
        return self._retrieve(
            query_text_feats,
            self.txt_queue,
            self.img_queue,
            k,
            refiner_node=refiner_node,
            query_ids=query_ids,
            current_epoch=current_epoch,
        )

    def diagnostics(self) -> Dict[str, float]:
        """Return JSON/log-friendly cumulative diagnostics."""
        valid = slice(0, self.size)
        result = {
            "mode": self.mode,
            "size": int(self.size),
            "capacity": int(self.queue_size),
            "unique_ids": (
                int(torch.unique(self.record_ids[valid]).numel())
                if self.mode != "fifo" and self.size
                else None
            ),
            "mean_quality": (
                float(self.quality[valid].mean().item())
                if self.mode != "fifo" and self.size
                else None
            ),
            "enqueue_rows": int(self._stats["enqueue_rows"]),
            "inserted_rows": int(self._stats["inserted_rows"]),
            "updated_rows": int(self._stats["updated_rows"]),
            "evicted_rows": int(self._stats["evicted_rows"]),
            "retrieval_calls": int(self._stats["retrieval_calls"]),
            "retrieval_queries": int(self._stats["retrieval_queries"]),
            "self_matches_masked": int(self._stats["self_matches_masked"]),
            "insufficient_support_calls": int(
                self._stats["insufficient_support_calls"]
            ),
        }
        metadata_count = self._stats["selected_metadata_count"]
        overlap_count = self._stats["raw_topk_overlap_count"]
        pair_count = self._stats["selected_pair_similarity_count"]
        result["selected_quality_mean"] = (
            self._stats["selected_quality_sum"] / metadata_count
            if metadata_count
            else None
        )
        result["selected_freshness_mean"] = (
            self._stats["selected_freshness_sum"] / metadata_count
            if metadata_count
            else None
        )
        result["raw_topk_overlap"] = (
            self._stats["raw_topk_overlap_sum"] / overlap_count
            if overlap_count
            else None
        )
        result["selected_pair_similarity_mean"] = (
            self._stats["selected_pair_similarity_sum"] / pair_count
            if pair_count
            else None
        )
        return result

    def state_dict(self) -> Dict[str, object]:
        """Capture the complete runtime state for a future strict resume path."""
        return {
            "schema": 1,
            "mode": self.mode,
            "queue_size": self.queue_size,
            "dim": self.dim,
            "ptr": self.ptr,
            "size": self.size,
            "is_full": self.is_full,
            "img_queue": self.img_queue.detach().clone(),
            "txt_queue": self.txt_queue.detach().clone(),
            "record_ids": self.record_ids.detach().clone(),
            "quality": self.quality.detach().clone(),
            "last_seen": self.last_seen.detach().clone(),
            "update_count": self.update_count.detach().clone(),
            "stats": dict(self._stats),
        }

    @torch.no_grad()
    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore a state produced by :meth:`state_dict` with strict checks."""
        if state.get("schema") != 1:
            raise ValueError("unsupported memory-state schema")
        for key, expected in (
            ("mode", self.mode),
            ("queue_size", self.queue_size),
            ("dim", self.dim),
        ):
            if state.get(key) != expected:
                raise ValueError("memory-state {} mismatch".format(key))

        for name in (
            "img_queue",
            "txt_queue",
            "record_ids",
            "quality",
            "last_seen",
            "update_count",
        ):
            destination = getattr(self, name)
            source = state[name]
            if not isinstance(source, torch.Tensor) or source.shape != destination.shape:
                raise ValueError("invalid tensor state for {}".format(name))
            destination.copy_(source.to(device=self.device, dtype=destination.dtype))

        self.ptr = int(state["ptr"])
        self.size = int(state["size"])
        self.is_full = bool(state["is_full"])
        if not 0 <= self.ptr < self.queue_size:
            raise ValueError("invalid restored memory pointer")
        if not 0 <= self.size <= self.queue_size:
            raise ValueError("invalid restored memory size")
        if self.is_full != (self.size == self.queue_size):
            raise ValueError("inconsistent restored full flag")

        self._id_to_slot = {}
        if self.mode != "fifo":
            ids = self.record_ids[: self.size].tolist()
            if any(record_id < 0 for record_id in ids) or len(set(ids)) != len(ids):
                raise ValueError("restored ID-aware memory contains invalid/duplicate IDs")
            self._id_to_slot = {
                int(record_id): slot for slot, record_id in enumerate(ids)
            }
        self._stats = defaultdict(float, state.get("stats", {}))
