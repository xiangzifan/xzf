import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PrecompDataset
from memory_bank import MemoryQueue


def unit_rows(*rows):
    return F.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)


class TinyRefiner(nn.Module):
    """Minimal differentiable stand-in for the unchanged GraphRefiner API."""

    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def forward(self, neighbors, query_feats=None):
        self.last_query_feats = query_feats
        return F.normalize(self.projection(neighbors).mean(dim=1), dim=-1)


class MetadataRecordingRefiner(nn.Module):
    """Records the selected RESB metadata passed to a QASR-style refiner."""

    requires_support_metadata = True

    def forward(
        self,
        neighbors,
        query_feats=None,
        support_quality=None,
        support_freshness=None,
    ):
        self.last_query_feats = query_feats
        self.last_quality = support_quality
        self.last_freshness = support_freshness
        return F.normalize(neighbors.mean(dim=1), dim=-1)


class MemoryBankTests(unittest.TestCase):
    def test_subset_reconstruction_preserves_original_record_ids(self):
        captions = ["caption {}".format(index) for index in range(6)]
        images = np.zeros((6, 2, 3), dtype=np.float32)
        selected = np.array([True, False, True, False, False, True])
        dataset = PrecompDataset(
            captions,
            images,
            vocab=None,
            data_split="train",
            mode="labeled",
            pred=selected,
            probability=np.linspace(0.1, 0.9, 6),
        )
        self.assertEqual(dataset.record_ids.tolist(), [0, 2, 5])

    def test_fifo_wrap_and_retrieval_match_original_rule(self):
        bank = MemoryQueue(queue_size=4, dim=2, mode="fifo", device=torch.device("cpu"))
        img = unit_rows([1, 0], [0, 1], [-1, 0])
        txt = unit_rows([0, 1], [1, 0], [0, -1])
        bank.enqueue(img, txt)
        bank.enqueue(unit_rows([1, 1], [-1, 1]), unit_rows([1, -1], [-1, -1]))

        self.assertEqual(bank.current_size, 4)
        self.assertTrue(bank.is_full)
        self.assertEqual(bank.ptr, 1)
        query = unit_rows([1, 1])
        expected_index = torch.mm(query, bank.img_queue.t()).argmax(dim=1)
        expected = bank.txt_queue[expected_index]
        actual = bank.retrieve_nearest_text(query, k=1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_id_refresh_keeps_one_slot_and_updates_metadata(self):
        bank = MemoryQueue(
            queue_size=4,
            dim=2,
            mode="id_refresh",
            alpha_min=0.5,
            alpha_max=0.5,
            quality_momentum=0.5,
            device=torch.device("cpu"),
        )
        bank.enqueue(
            unit_rows([1, 0], [0, 1]),
            unit_rows([0, 1], [1, 0]),
            record_ids=[10, 20],
            clean_prob=[0.8, 0.7],
            current_epoch=0,
        )
        bank.enqueue(
            unit_rows([0, 1]),
            unit_rows([1, 0]),
            record_ids=[10],
            clean_prob=[1.0],
            current_epoch=2,
        )

        self.assertEqual(bank.current_size, 2)
        self.assertEqual(bank.diagnostics()["unique_ids"], 2)
        slot = bank._id_to_slot[10]
        expected = F.normalize(torch.tensor([0.5, 0.5]), dim=0)
        self.assertTrue(torch.allclose(bank.img_queue[slot], expected, atol=1e-6))
        self.assertAlmostEqual(float(bank.quality[slot]), 0.9, places=6)
        self.assertEqual(int(bank.last_seen[slot]), 2)
        self.assertEqual(int(bank.update_count[slot]), 2)

    def test_exact_query_record_is_excluded(self):
        bank = MemoryQueue(
            queue_size=4,
            dim=2,
            mode="id_refresh",
            device=torch.device("cpu"),
        )
        bank.enqueue(
            unit_rows([1, 0], [0.8, 0.2]),
            unit_rows([1, 0], [0, 1]),
            record_ids=[1, 2],
            clean_prob=[0.9, 0.9],
            current_epoch=0,
        )
        result = bank.retrieve_nearest_text(
            unit_rows([1, 0]), k=1, query_ids=[1], current_epoch=1
        )
        self.assertTrue(torch.allclose(result, unit_rows([0, 1]), atol=1e-6))
        self.assertEqual(bank.diagnostics()["self_matches_masked"], 1)

    def test_quality_reranking_can_change_the_first_support(self):
        bank = MemoryQueue(
            queue_size=4,
            dim=2,
            mode="quality_diverse",
            candidate_pool_size=2,
            quality_weight=2.0,
            freshness_weight=0.0,
            mmr_balance=1.0,
            device=torch.device("cpu"),
        )
        bank.enqueue(
            unit_rows([1, 0], [0.99, 0.10]),
            unit_rows([1, 0], [0, 1]),
            record_ids=[1, 2],
            clean_prob=[0.1, 1.0],
            current_epoch=0,
        )
        result = bank.retrieve_nearest_text(
            unit_rows([1, 0]), k=1, query_ids=[99], current_epoch=0
        )
        self.assertTrue(torch.allclose(result, unit_rows([0, 1]), atol=1e-6))

    def test_mmr_prefers_a_nonredundant_value(self):
        bank = MemoryQueue(
            queue_size=5,
            dim=3,
            mode="quality_diverse",
            candidate_pool_size=3,
            quality_weight=0.0,
            freshness_weight=0.0,
            mmr_balance=0.30,
            device=torch.device("cpu"),
        )
        bank.enqueue(
            unit_rows([1, 0, 0], [0.99, 0.10, 0], [0.98, 0, 0.20]),
            unit_rows([1, 0, 0], [1, 0, 0], [0, 1, 0]),
            record_ids=[1, 2, 3],
            clean_prob=[1.0, 1.0, 1.0],
            current_epoch=0,
        )
        selected, indices = bank._select_supports(
            unit_rows([1, 0, 0]),
            bank.img_queue,
            bank.txt_queue,
            k=2,
            query_ids=[99],
            current_epoch=0,
        )
        self.assertEqual(indices.tolist(), [[0, 2]])
        self.assertTrue(torch.allclose(selected[0, 1], unit_rows([0, 1, 0])[0]))

    def test_state_roundtrip_preserves_id_mapping_and_output(self):
        kwargs = dict(
            queue_size=4,
            dim=2,
            mode="quality_diverse",
            candidate_pool_size=2,
            device=torch.device("cpu"),
        )
        source = MemoryQueue(**kwargs)
        source.enqueue(
            unit_rows([1, 0], [0, 1]),
            unit_rows([0, 1], [1, 0]),
            record_ids=[7, 8],
            clean_prob=[0.8, 0.9],
            current_epoch=1,
        )
        restored = MemoryQueue(**kwargs)
        state = source.state_dict()
        self.assertEqual(state["schema"], 2)
        self.assertEqual(tuple(state["img_queue"].shape), (2, 2))
        restored.load_state_dict(state)

        self.assertEqual(restored._id_to_slot, source._id_to_slot)
        query = unit_rows([1, 0])
        expected = source.retrieve_nearest_text(
            query, k=1, query_ids=[99], current_epoch=2
        )
        actual = restored.retrieve_nearest_text(
            query, k=1, query_ids=[99], current_epoch=2
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_selected_supports_keep_graph_refiner_gradient_path(self):
        bank = MemoryQueue(
            queue_size=4,
            dim=4,
            mode="quality_diverse",
            candidate_pool_size=3,
            device=torch.device("cpu"),
        )
        features = F.normalize(torch.eye(4), dim=-1)
        bank.enqueue(
            features,
            features.roll(shifts=1, dims=0),
            record_ids=[1, 2, 3, 4],
            clean_prob=[0.9, 0.8, 0.7, 0.6],
            current_epoch=0,
        )
        refiner = TinyRefiner(dim=4)
        query = features[:2]
        output = bank.retrieve_nearest_text(
            query,
            k=2,
            refiner_node=refiner,
            query_ids=[99, 100],
            current_epoch=1,
        )
        output.sum().backward()
        self.assertIs(refiner.last_query_feats, query)
        self.assertIsNotNone(refiner.projection.weight.grad)
        self.assertTrue(torch.isfinite(refiner.projection.weight.grad).all())

    def test_selected_support_metadata_is_aligned_with_qasr_values(self):
        bank = MemoryQueue(
            queue_size=4,
            dim=3,
            mode="quality_diverse",
            candidate_pool_size=3,
            quality_weight=0.0,
            freshness_weight=0.0,
            mmr_balance=1.0,
            freshness_tau=2.0,
            device=torch.device("cpu"),
        )
        keys = unit_rows([1, 0, 0], [0.8, 0.6, 0], [0, 1, 0])
        values = unit_rows([0, 1, 0], [0, 0, 1], [1, 0, 0])
        bank.enqueue(
            keys,
            values,
            record_ids=[1, 2, 3],
            clean_prob=[0.9, 0.7, 0.5],
            current_epoch=1,
        )
        refiner = MetadataRecordingRefiner()
        query = unit_rows([1, 0, 0])
        bank.retrieve_nearest_text(
            query,
            k=2,
            refiner_node=refiner,
            query_ids=[99],
            current_epoch=3,
        )
        selected = bank._select_supports(
            query,
            bank.img_queue,
            bank.txt_queue,
            k=2,
            query_ids=[99],
            current_epoch=3,
        )
        _, selected_indices = selected
        expected_quality = bank.quality[selected_indices]
        expected_freshness = torch.exp(
            -(3 - bank.last_seen[selected_indices]).float() / bank.freshness_tau
        )
        self.assertIs(refiner.last_query_feats, query)
        self.assertTrue(torch.equal(refiner.last_quality, expected_quality))
        self.assertTrue(
            torch.allclose(refiner.last_freshness, expected_freshness, atol=1e-7)
        )


if __name__ == "__main__":
    unittest.main()
