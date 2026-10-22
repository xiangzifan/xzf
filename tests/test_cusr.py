import unittest

import torch

from selective_rectification import (
    CUSRDiagnostics,
    bidirectional_retrieval_utility,
    build_cusr_evidence,
    legacy_infonce_per_sample,
    selective_infonce_loss,
)


class CUSRTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def test_two_sample_utility_matches_closed_form(self):
        scores = torch.tensor([[0.9, 0.2], [0.3, 0.8]])
        utility = bidirectional_retrieval_utility(scores, temperature=0.05)
        expected = torch.tensor([
            0.9 - 0.5 * (0.2 + 0.3),
            0.8 - 0.5 * (0.3 + 0.2),
        ])
        self.assertTrue(torch.allclose(utility, expected, atol=1e-7))

    def test_per_sample_loss_mean_matches_legacy_formula(self):
        scores = torch.randn(7, 7) * 0.2
        tau = 0.05
        eps = 1e-7
        exponentiated = (scores / tau).exp()
        i2t = exponentiated / exponentiated.sum(dim=1, keepdim=True)
        t2i = exponentiated.t() / exponentiated.t().sum(dim=1, keepdim=True)
        mask = torch.eye(scores.size(0)) + eps
        expected = 0.5 * (
            -(mask.log() * i2t).sum(dim=1).mean()
            -(mask.log() * t2i).sum(dim=1).mean()
            -i2t.diagonal().log().mean()
            -t2i.diagonal().log().mean()
        )
        actual = legacy_infonce_per_sample(scores, tau=tau).mean()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_evidence_selects_only_beneficial_positive_utility_pair(self):
        observed = torch.tensor([[0.2, 0.0], [0.0, 0.8]], requires_grad=True)
        prototype = torch.tensor([[0.9, 0.0], [0.7, 0.1]], requires_grad=True)
        evidence = build_cusr_evidence(
            observed,
            prototype,
            clean_probability=torch.tensor([0.1, 0.1]),
            utility_temperature=0.05,
            gate_temperature=0.1,
        )
        self.assertEqual(evidence["selected"].tolist(), [True, False])
        self.assertGreater(float(evidence["weights"][0]), 0.45)
        self.assertEqual(float(evidence["weights"][1]), 0.0)
        for value in evidence.values():
            self.assertFalse(value.requires_grad)

    def test_weighted_loss_routes_gradient_only_through_prototype_scores(self):
        observed = torch.tensor(
            [[0.2, 0.0], [0.0, 0.8]], requires_grad=True
        )
        prototype = torch.tensor(
            [[0.9, 0.0], [0.7, 0.1]], requires_grad=True
        )
        evidence = build_cusr_evidence(
            observed,
            prototype,
            clean_probability=torch.tensor([0.1, 0.1]),
            gate_temperature=0.1,
        )
        loss, _, weight_sum = selective_infonce_loss(
            prototype, evidence["weights"], scale=0.5
        )
        self.assertGreater(float(weight_sum), 0.0)
        loss.backward()
        self.assertIsNone(observed.grad)
        self.assertIsNotNone(prototype.grad)
        self.assertTrue(torch.isfinite(prototype.grad).all())

    def test_zero_selection_returns_finite_differentiable_zero(self):
        scores = torch.randn(3, 3, requires_grad=True)
        loss, per_sample, weight_sum = selective_infonce_loss(
            scores, torch.zeros(3)
        )
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(weight_sum), 0.0)
        self.assertTrue(torch.isfinite(per_sample).all())
        loss.backward()
        self.assertTrue(torch.equal(scores.grad, torch.zeros_like(scores)))

    def test_singleton_batch_is_never_selected(self):
        evidence = build_cusr_evidence(
            torch.tensor([[0.1]]),
            torch.tensor([[0.9]]),
            clean_probability=torch.tensor([0.0]),
        )
        self.assertFalse(bool(evidence["selected"].item()))
        self.assertEqual(float(evidence["weights"].item()), 0.0)

    def test_diagnostics_use_ground_truth_for_audit_only(self):
        observed = torch.tensor([[0.2, 0.0], [0.0, 0.8]])
        prototype = torch.tensor([[0.9, 0.0], [0.7, 0.1]])
        evidence = build_cusr_evidence(
            observed,
            prototype,
            clean_probability=torch.tensor([0.1, 0.1]),
            gate_temperature=0.1,
        )
        diagnostics = CUSRDiagnostics()
        diagnostics.update(evidence, clean_labels=[0, 1])
        result = diagnostics.as_dict()
        self.assertEqual(result["queries"], 2)
        self.assertAlmostEqual(result["accept_rate"], 0.5)
        self.assertAlmostEqual(result["audit_noisy_precision"], 1.0)
        self.assertAlmostEqual(result["audit_noisy_coverage"], 1.0)
        self.assertAlmostEqual(result["audit_clean_false_correction_rate"], 0.0)

    def test_invalid_probability_and_temperature_are_rejected(self):
        scores = torch.eye(2)
        with self.assertRaises(ValueError):
            build_cusr_evidence(scores, scores, [0.2, 1.1])
        with self.assertRaises(ValueError):
            bidirectional_retrieval_utility(scores, temperature=0.0)
        with self.assertRaises(ValueError):
            selective_infonce_loss(scores, [1.0, -0.1])


if __name__ == "__main__":
    unittest.main()
