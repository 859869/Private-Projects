"""
LLM-Generated Rule-Based Neighbor Pruner
Domain-specific pruning using interpretable, zero-shot rules
"""
import numpy as np
import math
import re
from collections import Counter
from functools import lru_cache
from html import unescape
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from .domain_rules import get_domain_rules


class LLMRulePruner:
    """
    LLM-generated rule-based pruner for domain-specific neighbor selection
    
    Key features:
    - Domain-adaptive: Different rules for Books, Movies, Restaurants, etc.
    - Interpretable: Clear semantic rules with explicit thresholds
    - Zero-shot: No training data required
    - Efficient: Simple rule evaluation at inference time
    """

    _STOP_WORDS = frozenset({
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'in', 'into', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this',
        'to', 'was', 'were', 'with', 'you', 'your', 'nan', 'none', 'null',
    })
    
    def __init__(
        self,
        dataset_name: str,
        k: int = 16,
        dataset=None,
        mix_min_users: int = 4,
        mix_min_items: int = 6,
        memory_storage=None,
    ):
        """
        Initialize LLM rule pruner
        
        Args:
            dataset_name: Name of dataset (e.g., 'instructrec-books')
            k: Total number of neighbors to select
            dataset: Dataset instance (for metadata access)
            memory_storage: MemoryStorage instance (for current user/item memories)
            mix_min_users: Minimum number of user neighbors
            mix_min_items: Minimum number of item neighbors
        """
        self.dataset_name = dataset_name
        self.k = k
        self.dataset = dataset
        self.memory_storage = memory_storage
        self.mix_min_users = mix_min_users
        self.mix_min_items = mix_min_items
        
        # Get domain-specific rules
        self.rules_class = get_domain_rules(dataset_name)
        print(f"  LLMRulePruner: Using {self.rules_class.__name__} for {dataset_name}")
        print(f"  Rules: {self.rules_class.get_description()}")

    @classmethod
    def _text_features(cls, text: str) -> Counter:
        """Build a small lexical feature vector from metadata or memory text."""
        if not text:
            return Counter()
        cleaned = re.sub(r'<[^>]+>', ' ', unescape(str(text))).lower()
        return Counter(
            token for token in re.findall(r"[a-z0-9]+", cleaned)
            if len(token) > 1 and token not in cls._STOP_WORDS
        )

    @staticmethod
    def _cosine_similarity(left: Counter, right: Counter) -> float:
        if not left or not right:
            return 0.0
        dot = sum(count * right.get(token, 0) for token, count in left.items())
        left_norm = math.sqrt(sum(count * count for count in left.values()))
        right_norm = math.sqrt(sum(count * count for count in right.values()))
        return dot / (left_norm * right_norm) if dot else 0.0

    @lru_cache(maxsize=20000)
    def _item_text_features(self, item_id: int) -> Counter:
        metadata = getattr(self.dataset, 'item_metadata', None) or {}
        item = metadata.get(item_id, {})
        title = item.get('title', '')
        description = item.get('description', '')
        # Metadata descriptions may contain long review lists; bound work per item.
        return self._text_features(f"{title} {title} {str(description)[:300]}")

    def _metadata_overlap(self, item_id: int, history_item_ids: List[int]) -> float:
        """Compare an item with the user's other recent history items."""
        item_features = self._item_text_features(item_id)
        if not item_features:
            return 0.0
        return max(
            (self._cosine_similarity(item_features, self._item_text_features(other_id))
             for other_id in history_item_ids[-20:] if other_id != item_id),
            default=0.0,
        )
    
    def extract_features(
        self,
        neighbor: Dict,
        user_memory: Optional[str] = None,
        neighbor_memory: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Extract features for a single neighbor
        
        Args:
            neighbor: Neighbor dict with keys:
                - type: 'user' or 'item'
                - id: neighbor ID
                - score: base CF score (edge_weight)
                - timestamp: last interaction time (optional)
                - metadata: item metadata (optional)
            user_memory: User's memory string
            neighbor_memory: Neighbor's memory string
        
        Returns:
            features: Dict with standardized feature names
        """
        features = {}
        
        # Base CF score (edge_weight)
        features['edge_weight'] = neighbor.get('score', 0.0)
        
        # Neighbor type
        features['neighbor_type'] = neighbor.get('type', 'item')
        
        # Co-interaction count (if available)
        features['co_interaction_count'] = neighbor.get('co_count', 0)
        
        # Recency (days since last interaction)
        timestamp = neighbor.get('timestamp', 0)
        if timestamp > 0:
            # Assume timestamp is days ago (or convert from actual timestamp)
            features['recency_days'] = timestamp
        else:
            features['recency_days'] = 0
        
        # Metadata overlap score (content similarity)
        features['metadata_overlap_score'] = neighbor.get('metadata_overlap', 0.0)
        
        # Memory similarity score
        if user_memory and neighbor_memory:
            features['memory_similarity_score'] = self._cosine_similarity(
                self._text_features(user_memory), self._text_features(neighbor_memory)
            )
        else:
            features['memory_similarity_score'] = neighbor.get('memory_sim', 0.0)
        
        return features
    
    def score_neighbor(
        self,
        neighbor: Dict,
        user_memory: Optional[str] = None,
        neighbor_memory: Optional[str] = None,
    ) -> float:
        """
        Score a single neighbor using domain-specific rules
        
        Args:
            neighbor: Neighbor dict
            user_memory: User's memory string
            neighbor_memory: Neighbor's memory string
        
        Returns:
            score: Final weighted score
        """
        # Extract features
        features = self.extract_features(neighbor, user_memory, neighbor_memory)
        
        # Apply domain-specific rules
        score = self.rules_class.apply_rules(features)
        
        return score
    
    def prune(
        self,
        user_id: int = None,
        graph = None,
        candidates: List[int] = None,
        user_memory: Optional[str] = None,
        neighbor_memories: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """
        Select top-k neighbors using domain-specific rules
        (Compatible with NeighborPruner interface)
        
        Args:
            user_id: User ID
            graph: UserItemGraph instance
            candidates: Optional list of candidate item IDs
            user_memory: Optional override for the user's stored memory
            neighbor_memories: Optional mapping of (type, id) to memory text
        
        Returns:
            pruned_result: Dict with 'neighbors' key containing selected neighbors
        """
        if graph is None:
            return {'neighbors': []}
        
        # Get neighbors from graph
        neighbor_data = graph.get_user_neighbors(user_id, max_items=100, max_users=50)
        history_item_ids = graph.get_user_items(user_id)
        if user_memory is None and self.memory_storage is not None:
            user_memory = self.memory_storage.get_user_memory(user_id)
        
        # Convert to uniform format
        candidate_neighbors = []
        
        # Add item neighbors
        for item_id, recency in neighbor_data['item_neighbors']:
            metadata_overlap = self._metadata_overlap(item_id, history_item_ids)
            
            # Use recency as edge weight for items (0-1 score)
            edge_weight = recency if recency else 0.5
            
            # Convert recency to days (approximate: assume higher recency = more recent = lower days)
            # recency is 0-1, where 1 = most recent
            # Approximate: 1.0 = 0 days, 0.0 = 365 days
            days_ago = (1.0 - recency) * 365 if recency else 180
            
            candidate_neighbors.append({
                'type': 'item',
                'id': item_id,
                'score': edge_weight,
                'metadata_overlap': metadata_overlap,
                'co_count': 0,  # Not applicable for items
                'timestamp': days_ago,
            })
        
        # Add user neighbors
        for neighbor_user_id, overlap_count in neighbor_data['user_neighbors']:
            # Use normalized overlap as edge weight for users
            user_degree = graph.get_user_degree(user_id)
            edge_weight = overlap_count / user_degree if user_degree > 0 else 0.5
            
            candidate_neighbors.append({
                'type': 'user',
                'id': neighbor_user_id,
                'score': edge_weight,
                'metadata_overlap': 0.0,  # Not applicable for users
                'co_count': overlap_count,
                'timestamp': 0,  # Users don't have recency
            })
        
        if not candidate_neighbors:
            return {'neighbors': []}
        
        # Score all neighbors
        scored_neighbors = []
        user_memory_features = self._text_features(user_memory)
        for neighbor in candidate_neighbors:
            neighbor_type, neighbor_id = neighbor['type'], neighbor['id']
            neighbor_memory = (neighbor_memories or {}).get((neighbor_type, neighbor_id))
            if neighbor_memory is None and self.memory_storage is not None:
                if neighbor_type == 'item':
                    neighbor_memory = self.memory_storage.get_item_memory(neighbor_id)
                else:
                    neighbor_memory = self.memory_storage.get_user_memory(neighbor_id)
            neighbor['memory_sim'] = self._cosine_similarity(
                user_memory_features, self._text_features(neighbor_memory)
            )
            score = self.score_neighbor(neighbor)
            scored_neighbors.append((score, neighbor))
        
        # Sort by score (descending)
        scored_neighbors.sort(key=lambda x: x[0], reverse=True)
        
        # Apply mixing constraints
        user_neighbors = [(s, n) for s, n in scored_neighbors if n['type'] == 'user']
        item_neighbors = [(s, n) for s, n in scored_neighbors if n['type'] == 'item']
        
        selected = []
        
        # Ensure minimum user neighbors
        selected.extend([n for _, n in user_neighbors[:self.mix_min_users]])
        
        # Ensure minimum item neighbors
        selected.extend([n for _, n in item_neighbors[:self.mix_min_items]])
        
        # Fill remaining slots with top-scored neighbors
        remaining_slots = self.k - len(selected)
        if remaining_slots > 0:
            selected_ids = {(n['type'], n['id']) for n in selected}
            for score, neighbor in scored_neighbors:
                if (neighbor['type'], neighbor['id']) not in selected_ids:
                    selected.append(neighbor)
                    remaining_slots -= 1
                    if remaining_slots == 0:
                        break
        
        # Trim to k
        selected = selected[:self.k]
        
        # Count items and users
        n_items = sum(1 for n in selected if n['type'] == 'item')
        n_users = sum(1 for n in selected if n['type'] == 'user')
        
        return {
            'user_id': user_id,
            'neighbors': selected,
            'n_items': n_items,
            'n_users': n_users
        }
    
    def get_rule_explanation(self, neighbor: Dict) -> str:
        """
        Get a human-readable explanation of why a neighbor was scored
        
        Args:
            neighbor: Neighbor dict
        
        Returns:
            explanation: Text explanation of scoring
        """
        features = self.extract_features(neighbor)
        score = self.rules_class.apply_rules(features)
        
        explanation = f"Neighbor {neighbor['type']}-{neighbor['id']}:\n"
        explanation += f"  Base score: {features['edge_weight']:.3f}\n"
        explanation += f"  Metadata overlap: {features['metadata_overlap_score']:.3f}\n"
        explanation += f"  Co-interactions: {features['co_interaction_count']}\n"
        explanation += f"  Memory similarity: {features['memory_similarity_score']:.3f}\n"
        explanation += f"  Recency (days): {features['recency_days']}\n"
        explanation += f"  Final score: {score:.3f}\n"
        explanation += f"  Rules: {self.rules_class.get_description()}\n"
        
        return explanation


class LLMRulePrunerAdapter:
    """
    Adapter to make LLMRulePruner compatible with existing MemRec code
    """
    
    def __init__(
        self,
        dataset_name: str,
        k: int = 16,
        dataset=None,
        **kwargs
    ):
        """Initialize adapter with LLM rule pruner"""
        self.pruner = LLMRulePruner(
            dataset_name=dataset_name,
            k=k,
            dataset=dataset,
            mix_min_users=kwargs.get('mix_min_users', 4),
            mix_min_items=kwargs.get('mix_min_items', 6),
            memory_storage=kwargs.get('memory_storage'),
        )
        self.k = k
    
    def prune(
        self,
        subgraph: Dict,
        user_memory: Optional[str] = None,
        neighbor_memories: Optional[Dict] = None,
    ) -> Dict:
        """
        Prune subgraph using LLM rules (compatible interface)
        
        Args:
            subgraph: Dict with 'neighbors' key containing list of neighbor dicts
            user_memory: User memory string
            neighbor_memories: Dict of neighbor memories
        
        Returns:
            pruned_subgraph: Dict with pruned 'neighbors' list
        """
        neighbors = subgraph.get('neighbors', [])
        
        # Prune using LLM rules
        pruned = self.pruner.prune(
            candidate_neighbors=neighbors,
            user_memory=user_memory,
            neighbor_memories=neighbor_memories,
        )
        
        # Return in same format
        return {
            'user_id': subgraph.get('user_id'),
            'neighbors': pruned,
            'pruning_method': 'llm_rules',
            'rule_class': self.pruner.rules_class.__name__,
        }

