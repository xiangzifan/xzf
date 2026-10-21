import unittest
from unittest import mock
import sys
import types
import os
import tempfile

import torch
import torch.nn.functional as F

# The local Windows validation environment has a torch/torchtext binary mismatch.
# GraphRefiner itself does not depend on VSE, so isolate that unrelated optional
# import while exercising the production GraphRefiner implementation in co_train.
vse_stub = types.ModuleType("vse")
vse_stub.VSE = object
vse_stub.infonce = lambda similarities: similarities.sum()
sys.modules.setdefault("vse", vse_stub)

from co_train import (
    GraphRefiner,
    ensure_rectification_modules,
    load_rectification_state_dict,
    rectification_state_dict,
)
from utils import save_checkpoint


class DummyNet:
    def __init__(self, dim):
        self.img_enc = torch.nn.Linear(dim, dim)
        self.optimizer = torch.optim.Adam(self.img_enc.parameters(), lr=5e-4)


class GraphRefinerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_mean_mode_is_equivalent_to_legacy_forward(self):
        refiner = GraphRefiner(
            dim=8, k=3, n_heads=2, dropout=0.0, mode="mean"
        ).eval()
        neighbors = torch.randn(4, 3, 8)

        with torch.no_grad():
            actual = refiner(neighbors)
            batch_size, support_count, dim = neighbors.shape
            projected_queries = refiner.q_proj(neighbors).view(
                batch_size, support_count, refiner.n_heads, refiner.d_k
            ).transpose(1, 2)
            projected_keys = refiner.k_proj(neighbors).view(
                batch_size, support_count, refiner.n_heads, refiner.d_k
            ).transpose(1, 2)
            projected_values = refiner.v_proj(neighbors).view(
                batch_size, support_count, refiner.n_heads, refiner.d_k
            ).transpose(1, 2)
            logits = torch.matmul(
                projected_queries, projected_keys.transpose(-2, -1)
            ) / (refiner.d_k ** 0.5)
            weights = F.softmax(logits, dim=-1)
            context = torch.matmul(weights, projected_values)
            context = context.transpose(1, 2).contiguous().view(
                batch_size, support_count, dim
            )
            nodes = refiner.layer_norm(
                neighbors + refiner.output_layer(context)
            )
            expected = F.normalize(nodes.mean(dim=1), p=2, dim=-1)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))

    def test_discriminative_utility_matches_two_query_closed_form(self):
        refiner = GraphRefiner(
            dim=2,
            k=2,
            n_heads=1,
            dropout=0.0,
            mode="dua_pool",
            utility_temperature=0.1,
        )
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        supports = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        )
        utility = refiner.compute_discriminative_utility(queries, supports)
        expected = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
        self.assertTrue(torch.allclose(utility, expected, atol=1e-6))

    def test_batch_size_one_has_finite_positive_similarity_fallback(self):
        refiner = GraphRefiner(
            dim=3, k=2, n_heads=1, dropout=0.0, mode="tugr_full"
        )
        query = torch.tensor([[1.0, 0.0, 0.0]])
        supports = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        )
        utility = refiner.compute_discriminative_utility(query, supports)
        self.assertTrue(torch.isfinite(utility).all())
        self.assertTrue(
            torch.allclose(utility, torch.tensor([[1.0, 0.0]]), atol=1e-6)
        )

    def test_all_factorial_modes_return_unit_norm_outputs(self):
        queries = F.normalize(torch.randn(5, 8), dim=-1)
        supports = F.normalize(torch.randn(5, 3, 8), dim=-1)
        for mode in ("mean", "dua_pool", "qdmp_mean", "tugr_full"):
            refiner = GraphRefiner(
                dim=8, k=3, n_heads=2, dropout=0.0, mode=mode
            )
            output = refiner(
                supports, query_feats=queries if mode != "mean" else None
            )
            self.assertEqual(tuple(output.shape), (5, 8))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(
                torch.allclose(output.norm(dim=-1), torch.ones(5), atol=1e-5)
            )
            diagnostics = refiner.diagnostics()
            self.assertEqual(diagnostics["calls"], 1)
            self.assertEqual(diagnostics["queries"], 5)
            self.assertLess(diagnostics["pool_sum_error"], 1e-6)

    def test_qdmp_and_dua_parameters_receive_finite_gradients(self):
        refiner = GraphRefiner(
            dim=8,
            k=3,
            n_heads=2,
            dropout=0.0,
            mode="tugr_full",
            detach_utility=True,
        )
        queries = F.normalize(torch.randn(4, 8), dim=-1)
        supports = F.normalize(torch.randn(4, 3, 8), dim=-1)
        loss = refiner(supports, query_feats=queries).square().sum(dim=-1).mean()
        # Unit norms make the exact squared-norm loss constant. Use a directional
        # objective so the test exercises the complete differentiable path.
        loss = loss + refiner(supports, query_feats=queries)[:, 0].sum()
        loss.backward()
        trainable = [
            parameter
            for parameter in refiner.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(parameter.grad is not None for parameter in trainable))
        for parameter in trainable:
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        for parameter in refiner.edge_nn.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)

    def test_utility_modes_require_aligned_query_tensor(self):
        refiner = GraphRefiner(
            dim=8, k=3, n_heads=2, dropout=0.0, mode="tugr_full"
        )
        supports = torch.randn(2, 3, 8)
        with self.assertRaises(ValueError):
            refiner(supports)
        with self.assertRaises(ValueError):
            refiner(supports, query_feats=torch.randn(3, 8))

    def test_runtime_rectification_state_roundtrip(self):
        options = types.SimpleNamespace(
            embed_size=4,
            k=2,
            learning_rate=5e-4,
            refiner_heads=2,
            refiner_dropout=0.0,
            refiner_mode="tugr_full",
            refiner_utility_temperature=0.1,
            refiner_pool_temperature=0.1,
            refiner_message_prior_strength=0.5,
            refiner_end_to_end_utility=False,
            refiner_eps=1e-6,
            queue_size=8,
            memory_mode="quality_diverse",
            memory_candidate_pool_size=3,
            memory_quality_weight=0.25,
            memory_freshness_weight=0.1,
            memory_freshness_tau=2.0,
            memory_mmr_balance=0.8,
            memory_alpha_min=0.2,
            memory_alpha_max=0.8,
            memory_quality_momentum=0.5,
        )
        source = DummyNet(dim=4)
        # Simulate a CUDA-capable host while keeping this fixture deliberately
        # on CPU.  Runtime construction must follow the encoder device instead
        # of MemoryQueue's CUDA-availability fallback.
        with mock.patch("memory_bank.torch.cuda.is_available", return_value=True):
            ensure_rectification_modules(options, source)
        self.assertEqual(
            source.memory_queue.device,
            next(source.img_enc.parameters()).device,
        )
        features = F.normalize(torch.eye(4), dim=-1)
        source.memory_queue.enqueue(
            features,
            features.roll(shifts=1, dims=0),
            record_ids=[1, 2, 3, 4],
            clean_prob=[0.9, 0.8, 0.7, 0.6],
            current_epoch=2,
        )
        state = rectification_state_dict(source)

        restored = DummyNet(dim=4)
        with mock.patch("memory_bank.torch.cuda.is_available", return_value=True):
            ensure_rectification_modules(options, restored)
        self.assertEqual(
            restored.memory_queue.device,
            next(restored.img_enc.parameters()).device,
        )
        load_rectification_state_dict(restored, state)

        self.assertEqual(
            restored.memory_queue._id_to_slot,
            source.memory_queue._id_to_slot,
        )
        for source_parameter, restored_parameter in zip(
            source.i2t_refiner_node.parameters(),
            restored.i2t_refiner_node.parameters(),
        ):
            self.assertTrue(
                torch.equal(source_parameter, restored_parameter)
            )

    def test_checkpoint_write_is_complete_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as directory:
            prefix = directory + os.sep
            save_checkpoint(
                {"value": torch.tensor([1.0, 2.0])},
                is_best=True,
                filename="checkpoint.pth.tar",
                prefix=prefix,
            )
            checkpoint_path = os.path.join(directory, "checkpoint.pth.tar")
            best_path = os.path.join(directory, "model_best.pth.tar")
            self.assertTrue(os.path.isfile(checkpoint_path))
            self.assertTrue(os.path.isfile(best_path))
            self.assertFalse(os.path.exists(checkpoint_path + ".tmp"))
            self.assertFalse(os.path.exists(best_path + ".tmp"))
            loaded = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.assertTrue(
                torch.equal(loaded["value"], torch.tensor([1.0, 2.0]))
            )


if __name__ == "__main__":
    unittest.main()
