"""Counterfactual utility-guided selective rectification (CUSR).

This module contains only tensor operations and diagnostic aggregation.  It is
kept separate from the training loop so the selection rule can be unit-tested
without constructing the full IN2R model.

The clean/noisy ground-truth label is accepted only by ``CUSRDiagnostics`` for
post-hoc auditing.  It is never consumed by the selection or loss functions.
"""

import math

import torch


def _validate_square_scores(scores, name):
    if scores.ndim != 2 or scores.size(0) != scores.size(1):
        raise ValueError("{} must be a square [batch, batch] tensor".format(name))
    if scores.size(0) < 1:
        raise ValueError("{} must contain at least one sample".format(name))
    if not torch.isfinite(scores).all():
        raise ValueError("{} must be finite".format(name))


def bidirectional_retrieval_utility(scores, temperature=0.05):
    """Return the negative-aware bidirectional utility of every diagonal pair.

    For a similarity matrix ``scores``, the utility is the diagonal similarity
    minus the mean of the query-to-target and target-to-query negative
    LogMeanExp terms.  LogMeanExp, rather than LogSumExp, makes the scale stable
    when the final mini-batch has a different size.

    A batch of one has no negatives.  Its diagonal similarity is returned for
    diagnostics, while ``build_cusr_evidence`` marks it ineligible for CUSR.
    """
    _validate_square_scores(scores, "scores")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    batch_size = scores.size(0)
    positive = scores.diagonal()
    if batch_size == 1:
        return positive

    negative_mask = ~torch.eye(
        batch_size, dtype=torch.bool, device=scores.device
    )
    negative_logits = scores.masked_fill(~negative_mask, -torch.inf)
    log_count = math.log(batch_size - 1)
    row_negative = temperature * (
        torch.logsumexp(negative_logits / temperature, dim=1) - log_count
    )
    column_negative = temperature * (
        torch.logsumexp(negative_logits / temperature, dim=0) - log_count
    )
    return positive - 0.5 * (row_negative + column_negative)


def build_cusr_evidence(
    observed_scores,
    prototype_scores,
    clean_probability,
    utility_temperature=0.05,
    gate_temperature=0.05,
    min_gain=0.0,
    min_utility=0.0,
):
    """Build detached CUSR evidence and sample weights.

    A sample is selected only when its memory prototype has positive absolute
    retrieval utility and improves over the observed correspondence.  The
    peer network's GMM clean probability controls how strongly an accepted
    sample is corrected.  All routing evidence is detached deliberately so the
    encoder cannot manipulate the selection rule as a gradient shortcut.
    """
    _validate_square_scores(observed_scores, "observed_scores")
    _validate_square_scores(prototype_scores, "prototype_scores")
    if observed_scores.shape != prototype_scores.shape:
        raise ValueError("observed_scores and prototype_scores must share a shape")
    if observed_scores.device != prototype_scores.device:
        raise ValueError("observed_scores and prototype_scores must share a device")
    if observed_scores.dtype != prototype_scores.dtype:
        raise ValueError("observed_scores and prototype_scores must share a dtype")
    if not math.isfinite(float(gate_temperature)) or gate_temperature <= 0:
        raise ValueError("gate_temperature must be finite and positive")
    if not math.isfinite(float(min_gain)) or not math.isfinite(float(min_utility)):
        raise ValueError("CUSR thresholds must be finite")

    batch_size = observed_scores.size(0)
    clean_probability = torch.as_tensor(
        clean_probability,
        dtype=observed_scores.dtype,
        device=observed_scores.device,
    ).reshape(-1)
    if clean_probability.numel() != batch_size:
        raise ValueError("clean_probability must contain one value per sample")
    if not torch.isfinite(clean_probability).all():
        raise ValueError("clean_probability must be finite")
    if bool(((clean_probability < 0) | (clean_probability > 1)).any().item()):
        raise ValueError("clean_probability must be in [0, 1]")

    with torch.no_grad():
        observed_utility = bidirectional_retrieval_utility(
            observed_scores.detach(), temperature=utility_temperature
        )
        prototype_utility = bidirectional_retrieval_utility(
            prototype_scores.detach(), temperature=utility_temperature
        )
        gain = prototype_utility - observed_utility
        selected = (gain > min_gain) & (prototype_utility > min_utility)
        if batch_size == 1:
            selected.zero_()
        noise_need = 1.0 - clean_probability
        confidence = torch.sigmoid(gain / gate_temperature)
        weights = selected.to(observed_scores.dtype) * noise_need * confidence

    return {
        "observed_utility": observed_utility,
        "prototype_utility": prototype_utility,
        "gain": gain,
        "selected": selected,
        "clean_probability": clean_probability.detach(),
        "noise_need": noise_need,
        "confidence": confidence,
        "weights": weights,
    }


def legacy_infonce_per_sample(scores, tau=0.05, eps=1e-7):
    """Exact per-pair decomposition of IN2R's existing ``infonce`` objective.

    The mean of the returned vector is algebraically equal to the scalar legacy
    loss.  The implementation intentionally preserves the original four terms
    instead of replacing the objective with conventional cross entropy.
    """
    _validate_square_scores(scores, "scores")
    if not math.isfinite(float(tau)) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    batch_size = scores.size(0)
    mask = torch.eye(
        batch_size, dtype=scores.dtype, device=scores.device
    ) + eps
    exponentiated = (scores / tau).exp()
    i2t = exponentiated / exponentiated.sum(dim=1, keepdim=True)
    t2i = exponentiated.t() / exponentiated.t().sum(dim=1, keepdim=True)
    log_mask = mask.log()

    cost_i2t_reverse = -(log_mask * i2t).sum(dim=1)
    cost_t2i_reverse = -(log_mask * t2i).sum(dim=1)
    cost_i2t = -i2t.diagonal().log()
    cost_t2i = -t2i.diagonal().log()
    return 0.5 * (
        cost_i2t_reverse + cost_t2i_reverse + cost_i2t + cost_t2i
    )


def selective_infonce_loss(scores, weights, tau=0.05, scale=0.5, eps=1e-12):
    """Return the normalized CUSR loss and its detached accounting values."""
    _validate_square_scores(scores, "scores")
    if not math.isfinite(float(scale)) or scale < 0:
        raise ValueError("scale must be finite and nonnegative")
    weights = torch.as_tensor(
        weights, dtype=scores.dtype, device=scores.device
    ).reshape(-1)
    if weights.numel() != scores.size(0):
        raise ValueError("weights must contain one value per sample")
    if not torch.isfinite(weights).all() or bool((weights < 0).any().item()):
        raise ValueError("weights must be finite and nonnegative")

    per_sample = legacy_infonce_per_sample(scores, tau=tau)
    weight_sum = weights.sum()
    if float(weight_sum.detach().item()) <= eps:
        loss = scores.sum() * 0.0
    else:
        loss = float(scale) * (weights * per_sample).sum() / weight_sum
    return loss, per_sample.detach(), weight_sum.detach()


class CUSRDiagnostics:
    """Aggregate train-visible CUSR evidence and audit-only ground truth."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = 0
        self.queries = 0
        self.accepted = 0
        self.positive_gain = 0
        self.positive_prototype_utility = 0
        self.zero_selected_batches = 0
        self.observed_utility_sum = 0.0
        self.prototype_utility_sum = 0.0
        self.gain_sum = 0.0
        self.clean_probability_sum = 0.0
        self.weight_sum = 0.0
        self.selected_weight_sum = 0.0
        self.audited_noisy = 0
        self.audited_clean = 0
        self.selected_noisy = 0
        self.selected_clean = 0

    def update(self, evidence, clean_labels=None):
        required = {
            "observed_utility",
            "prototype_utility",
            "gain",
            "selected",
            "clean_probability",
            "weights",
        }
        missing = sorted(required.difference(evidence))
        if missing:
            raise ValueError("missing CUSR evidence fields: {}".format(missing))

        values = {name: evidence[name].detach().reshape(-1) for name in required}
        count = values["gain"].numel()
        if count < 1 or any(value.numel() != count for value in values.values()):
            raise ValueError("CUSR evidence fields must have the same nonzero length")
        for name, value in values.items():
            if name != "selected" and not torch.isfinite(value).all():
                raise ValueError("CUSR diagnostic {} must be finite".format(name))

        selected = values["selected"].bool()
        accepted = int(selected.sum().item())
        self.calls += 1
        self.queries += count
        self.accepted += accepted
        self.positive_gain += int((values["gain"] > 0).sum().item())
        self.positive_prototype_utility += int(
            (values["prototype_utility"] > 0).sum().item()
        )
        self.zero_selected_batches += int(accepted == 0)
        self.observed_utility_sum += float(values["observed_utility"].sum().item())
        self.prototype_utility_sum += float(values["prototype_utility"].sum().item())
        self.gain_sum += float(values["gain"].sum().item())
        self.clean_probability_sum += float(values["clean_probability"].sum().item())
        self.weight_sum += float(values["weights"].sum().item())
        self.selected_weight_sum += float(values["weights"][selected].sum().item())

        if clean_labels is not None:
            labels = torch.as_tensor(clean_labels).reshape(-1).to(torch.long)
            if labels.numel() != count:
                raise ValueError("clean_labels must contain one value per sample")
            if bool(((labels != 0) & (labels != 1)).any().item()):
                raise ValueError("clean_labels audit values must be binary")
            selected_cpu = selected.cpu()
            labels = labels.cpu()
            noisy = labels == 0
            clean = labels == 1
            self.audited_noisy += int(noisy.sum().item())
            self.audited_clean += int(clean.sum().item())
            self.selected_noisy += int((selected_cpu & noisy).sum().item())
            self.selected_clean += int((selected_cpu & clean).sum().item())

    @staticmethod
    def _ratio(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else 0.0

    def as_dict(self):
        return {
            "calls": self.calls,
            "queries": self.queries,
            "accept_rate": self._ratio(self.accepted, self.queries),
            "positive_gain_rate": self._ratio(self.positive_gain, self.queries),
            "positive_prototype_utility_rate": self._ratio(
                self.positive_prototype_utility, self.queries
            ),
            "observed_utility_mean": self._ratio(
                self.observed_utility_sum, self.queries
            ),
            "prototype_utility_mean": self._ratio(
                self.prototype_utility_sum, self.queries
            ),
            "utility_gain_mean": self._ratio(self.gain_sum, self.queries),
            "clean_probability_mean": self._ratio(
                self.clean_probability_sum, self.queries
            ),
            "weight_mean_all": self._ratio(self.weight_sum, self.queries),
            "weight_mean_selected": self._ratio(
                self.selected_weight_sum, self.accepted
            ),
            "zero_selected_batches": self.zero_selected_batches,
            "audit_noisy_precision": self._ratio(
                self.selected_noisy, self.accepted
            ),
            "audit_noisy_coverage": self._ratio(
                self.selected_noisy, self.audited_noisy
            ),
            "audit_clean_false_correction_rate": self._ratio(
                self.selected_clean, self.audited_clean
            ),
        }
