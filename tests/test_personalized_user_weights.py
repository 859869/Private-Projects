import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.memory.graph import UserItemGraph
from src.memory.personalized_weights import UserNeighborFeatures, UserWeightModel
from src.memory.personalized_weights.training import build_pairs, fit_gate, pair_metrics
from src.memory.pruner_llm_rules import LLMRulePruner


class PersonalizedUserWeightTests(unittest.TestCase):
    def setUp(self):
        self.dataset = SimpleNamespace(
            train_data={10: [1, 2], 20: [1, 3], 30: [1, 4], 40: [1, 5]},
            item_metadata={i: {"title": title} for i, title in {
                1: "Shared Atlas", 2: "Space Atlas", 3: "Space Guide",
                4: "Garden Guide", 5: "Garden Recipes"}.items()},
        )
        self.graph = UserItemGraph(self.dataset)

    def test_inner_holdout_produces_trainable_pairs(self):
        targets = {10: 3, 20: 4, 30: 2, 40: 2}
        records, coverage = build_pairs(self.dataset, self.graph, targets, seed=1)
        self.assertGreaterEqual(coverage["covered_users"], 2)
        self.assertGreater(len(records), 0)
        global_model = fit_gate(records, records, personalized=False, epochs=5)
        user_model = fit_gate(records, records, personalized=True, epochs=5,
                              initial_model=global_model)
        self.assertIsNotNone(pair_metrics(user_model, records)["pair_loss"])
        self.assertTrue(np.allclose(global_model.coefficients, 0.0))

    def test_checkpoint_and_context_change_weights(self):
        coefficients = np.zeros((4, 3))
        coefficients[0] = [4.0, -4.0, 0.0]
        model = UserWeightModel(coefficients=coefficients)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            model.save(path)
            restored = UserWeightModel.load(path)
        low = restored.weights([0, 0, 0, 0])
        high = restored.weights([1, 0, 0, 0])
        self.assertAlmostEqual(float(high.sum()), 1.0)
        self.assertGreater(high[0], low[0])

    def test_model_changes_only_user_neighbor_scoring(self):
        coefficients = np.zeros((4, 3))
        coefficients[0] = [3.0, -3.0, 0.0]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.json"
            UserWeightModel(coefficients=coefficients).save(checkpoint)
            original = LLMRulePruner("instructrec-books", dataset=self.dataset,
                                     k=5, mix_min_users=1, mix_min_items=1)
            weighted = LLMRulePruner("instructrec-books", dataset=self.dataset,
                                     k=5, mix_min_users=1, mix_min_items=1,
                                     user_weight_checkpoint=str(checkpoint))
            before = original.prune(10, self.graph)
            after = weighted.prune(10, self.graph)
        original_items = {n["id"]: n["pruning_score"] for n in before["neighbors"]
                          if n["type"] == "item"}
        weighted_items = {n["id"]: n["pruning_score"] for n in after["neighbors"]
                          if n["type"] == "item"}
        self.assertEqual(original_items, weighted_items)
        self.assertEqual(len(after["user_neighbor_weights"]), 3)
        self.assertTrue(any(n["pruning_score"] != n["score"] for n in after["neighbors"]
                            if n["type"] == "user"))
        user_neighbor = next(n for n in after["neighbors"] if n["type"] == "user")
        self.assertEqual(weighted.score_neighbor(user_neighbor), user_neighbor["pruning_score"])


if __name__ == "__main__":
    unittest.main()
