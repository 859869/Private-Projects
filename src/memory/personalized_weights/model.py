"""A small, interpretable gate trained with pairwise neighbor ranking."""

import json
from pathlib import Path

import numpy as np

from .features import CONTEXT_NAMES, SIGNAL_NAMES


class UserWeightModel:
    """Fixed trained parameters produce a different signal mixture per user."""

    def __init__(self, global_logits=None, coefficients=None):
        self.global_logits = np.zeros(len(SIGNAL_NAMES), dtype=np.float64) if global_logits is None else np.asarray(global_logits, dtype=np.float64)
        self.coefficients = np.zeros((len(CONTEXT_NAMES), len(SIGNAL_NAMES)), dtype=np.float64) if coefficients is None else np.asarray(coefficients, dtype=np.float64)
        if self.global_logits.shape != (len(SIGNAL_NAMES),) or self.coefficients.shape != (len(CONTEXT_NAMES), len(SIGNAL_NAMES)):
            raise ValueError("Weight model dimensions do not match the feature schema")

    def weights(self, context):
        context = np.asarray(context, dtype=np.float64)
        if context.shape != (len(CONTEXT_NAMES),):
            raise ValueError("Invalid user context shape")
        logits = self.global_logits + context @ self.coefficients
        logits -= np.max(logits)
        exp = np.exp(logits)
        return exp / exp.sum()

    def score(self, context, base_score, signals, memory_similarity=0.0):
        """Keep the existing base scale and fixed memory boost."""
        signals = np.asarray(signals, dtype=np.float64)
        return float(base_score * (1.0 + 2.0 * self.weights(context) @ signals)
                     * (1.0 + 0.8 * max(0.0, memory_similarity)))

    def save(self, path, metadata=None):
        payload = {
            "version": 1,
            "context_names": list(CONTEXT_NAMES),
            "signal_names": list(SIGNAL_NAMES),
            "global_logits": self.global_logits.tolist(),
            "coefficients": self.coefficients.tolist(),
            "metadata": metadata or {},
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1 or payload.get("context_names") != list(CONTEXT_NAMES) or payload.get("signal_names") != list(SIGNAL_NAMES):
            raise ValueError("Incompatible user weight checkpoint schema")
        return cls(payload["global_logits"], payload["coefficients"])
