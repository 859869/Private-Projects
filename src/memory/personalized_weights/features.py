"""Features available before the recommendation being evaluated."""

import math
import re
from collections import Counter

import numpy as np


CONTEXT_NAMES = ("history_size", "candidate_count", "best_overlap", "mean_overlap")
SIGNAL_NAMES = ("overlap_fraction", "jaccard", "history_text_similarity")


class UserNeighborFeatures:
    """Build the same user and neighbor features during training and inference."""

    def __init__(self, graph, item_metadata=None):
        self.graph = graph
        self.item_metadata = item_metadata or {}
        self._profiles = {}

    def context(self, user_id, user_neighbors):
        degree = self.graph.get_user_degree(user_id)
        overlaps = [count / max(1, degree) for _, count in user_neighbors]
        return np.asarray([
            min(math.log1p(degree) / math.log1p(100), 1.0),
            min(len(user_neighbors) / 50.0, 1.0),
            max(overlaps, default=0.0),
            float(np.mean(overlaps)) if overlaps else 0.0,
        ], dtype=np.float64)

    def _profile(self, user_id):
        if user_id not in self._profiles:
            tokens = Counter()
            for item_id in self.graph.get_user_items(user_id)[-20:]:
                meta = self.item_metadata.get(item_id, {})
                # Titles are available independently of future user feedback.
                content = str(meta.get('title', ''))
                tokens.update(token for token in re.findall(r"[a-z0-9]+", content.lower())
                              if len(token) > 2)
            self._profiles[user_id] = tokens
        return self._profiles[user_id]

    def signals(self, user_id, neighbor_id, overlap_count):
        user_degree = self.graph.get_user_degree(user_id)
        neighbor_degree = self.graph.get_user_degree(neighbor_id)
        union = user_degree + neighbor_degree - overlap_count
        left, right = self._profile(user_id), self._profile(neighbor_id)
        dot = sum(value * right.get(token, 0) for token, value in left.items())
        norm_left = math.sqrt(sum(value * value for value in left.values()))
        norm_right = math.sqrt(sum(value * value for value in right.values()))
        similarity = dot / (norm_left * norm_right) if norm_left and norm_right else 0.0
        return np.asarray([
            overlap_count / max(1, user_degree),
            overlap_count / max(1, union),
            similarity,
        ], dtype=np.float64)
