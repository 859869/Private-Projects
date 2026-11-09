"""Offline pairwise training on an inner holdout of the training histories."""

import numpy as np

from .features import UserNeighborFeatures
from .model import UserWeightModel


def build_pairs(dataset, graph, targets, max_users=None, negatives_per_positive=5, seed=42):
    """Label a neighbor positive if its *prefix* contains the target's held-out item."""
    rng = np.random.default_rng(seed)
    extractor = UserNeighborFeatures(graph, getattr(dataset, "item_metadata", None))
    records = []
    stats = {"eligible_users": 0, "covered_users": 0, "pairs": 0}
    user_ids = list(targets)
    rng.shuffle(user_ids)
    for user_id in user_ids[:max_users]:
        neighbors = graph.get_user_neighbors(user_id, max_items=100, max_users=50)["user_neighbors"]
        if len(neighbors) < 2:
            continue
        stats["eligible_users"] += 1
        target = targets[user_id]
        positive = [(neighbor_id, overlap) for neighbor_id, overlap in neighbors
                    if target in graph.get_user_items(neighbor_id)]
        negative = [(neighbor_id, overlap) for neighbor_id, overlap in neighbors
                    if target not in graph.get_user_items(neighbor_id)]
        if not positive or not negative:
            continue
        stats["covered_users"] += 1
        context = extractor.context(user_id, neighbors)
        degree = max(1, graph.get_user_degree(user_id))
        for pos_id, pos_overlap in positive:
            chosen = rng.choice(len(negative), size=min(negatives_per_positive, len(negative)), replace=False)
            pos_signals = extractor.signals(user_id, pos_id, pos_overlap)
            pos_base = pos_overlap / degree
            for index in np.atleast_1d(chosen):
                neg_id, neg_overlap = negative[int(index)]
                neg_signals = extractor.signals(user_id, neg_id, neg_overlap)
                neg_base = neg_overlap / degree
                # score = base + weight dot (2 * base * signals)
                records.append((user_id, context, pos_base - neg_base,
                                2.0 * (pos_base * pos_signals - neg_base * neg_signals)))
    stats["pairs"] = len(records)
    return records, stats


def pair_metrics(model, records):
    if not records:
        return {"pair_accuracy": None, "pair_loss": None}
    contexts = np.stack([row[1] for row in records])
    base = np.asarray([row[2] for row in records])
    differences = np.stack([row[3] for row in records])
    logits = contexts @ model.coefficients + model.global_logits
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    margins = base + np.sum(weights * differences, axis=1)
    return {"pair_accuracy": float(np.mean(margins > 0)),
            "pair_loss": float(np.mean(np.logaddexp(0, -margins)))}


def fit_gate(train_records, valid_records, *, personalized, epochs=150, learning_rate=0.5,
             regularization=0.01, initial_model=None):
    if not train_records:
        raise ValueError("No trainable user-neighbor pairs; inspect coverage before training")
    model = UserWeightModel() if initial_model is None else UserWeightModel(
        initial_model.global_logits.copy(), initial_model.coefficients.copy())
    contexts = np.stack([row[1] for row in train_records])
    base = np.asarray([row[2] for row in train_records])
    differences = np.stack([row[3] for row in train_records])
    best_model = UserWeightModel(model.global_logits.copy(), model.coefficients.copy())
    best_loss = float("inf")
    for _ in range(epochs):
        logits = contexts @ model.coefficients + model.global_logits
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        weighted_diff = np.sum(weights * differences, axis=1)
        margins = base + weighted_diff
        sigmoid_negative = np.exp(-np.logaddexp(0, margins))
        grad_logits = -sigmoid_negative[:, None] * weights * (differences - weighted_diff[:, None])
        model.global_logits -= learning_rate * grad_logits.mean(axis=0)
        if personalized:
            gradient = contexts.T @ grad_logits / len(train_records)
            model.coefficients -= learning_rate * (gradient + regularization * model.coefficients)
        else:
            model.coefficients[:] = 0.0
        validation_loss = pair_metrics(model, valid_records or train_records)["pair_loss"]
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_model = UserWeightModel(model.global_logits.copy(), model.coefficients.copy())
    return best_model
