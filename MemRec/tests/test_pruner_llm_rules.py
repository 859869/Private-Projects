import unittest
from types import SimpleNamespace

from src.memory.graph import UserItemGraph
from src.memory.pruner_llm_rules import LLMRulePruner
from src.memory.storage import MemoryStorage


class LLMRulePrunerFeatureTests(unittest.TestCase):
    def setUp(self):
        self.dataset = SimpleNamespace(
            item_metadata={
                1: {'title': 'Galactic Atlas', 'description': 'A guide to stars and galaxies'},
                2: {'title': 'Galactic Atlas Volume Two', 'description': 'A guide to stars and galaxies'},
                3: {'title': 'Garden Cookbook', 'description': 'Recipes for fresh vegetables'},
                4: {'title': 'Galactic Atlas Companion', 'description': 'A guide to stars and galaxies'},
            },
            train_data={10: [1, 2, 3], 20: [2, 4]},
        )
        self.storage = MemoryStorage()
        self.storage.update_user_memory(10, 'Enjoys stars and galaxies')
        self.storage.update_user_memory(20, 'Reads about stars and galaxies')
        self.storage.initialize_item_descriptions(self.dataset.item_metadata)
        self.pruner = LLMRulePruner(
            'instructrec-books', dataset=self.dataset,
            memory_storage=self.storage, k=4,
        )

    def test_metadata_uses_other_history_items_and_excludes_self(self):
        self.assertEqual(self.pruner._metadata_overlap(1, [1]), 0.0)
        self.assertGreater(self.pruner._metadata_overlap(1, [1, 2, 3]), 0.6)
        self.assertLess(self.pruner._metadata_overlap(3, [1, 2, 3]), 0.6)

    def test_prune_uses_current_metadata_and_memories(self):
        neighbors = self.pruner.prune(10, UserItemGraph(self.dataset))['neighbors']
        similar_item = next(n for n in neighbors if n['type'] == 'item' and n['id'] == 2)
        similar_user = next(n for n in neighbors if n['type'] == 'user' and n['id'] == 20)
        self.assertGreater(similar_item['metadata_overlap'], 0.6)
        self.assertGreater(similar_item['memory_sim'], 0.0)
        self.assertGreater(similar_user['memory_sim'], 0.0)

    def test_missing_text_has_no_artificial_similarity(self):
        self.dataset.item_metadata = None
        pruner = LLMRulePruner('instructrec-books', dataset=self.dataset)
        self.assertEqual(pruner._metadata_overlap(1, [1, 2]), 0.0)
        self.assertEqual(
            pruner.extract_features({'type': 'item', 'id': 1})['memory_similarity_score'],
            0.0,
        )


if __name__ == '__main__':
    unittest.main()
