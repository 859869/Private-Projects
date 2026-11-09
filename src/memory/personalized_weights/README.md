# Personalized user-neighbor weights

This directory contains a small offline-trained weight gate. Its parameters stay
fixed at inference; each target user's graph statistics produce a different
mixture over three user-neighbor signals: overlap fraction, Jaccard overlap,
and similarity of recent book titles. Item-neighbor ranking remains on the
existing domain rule. LLM memory similarity keeps the original fixed boost
because historical LLM memory snapshots are unavailable for offline training.

## Train

From the repository root:

```powershell
python scripts/train_user_neighbor_weights.py --dataset instructrec-books --output results/user_neighbor_weights_books.json
```

The script writes both a personalized checkpoint and a `*.global.json`
checkpoint with the same features and zero user-dependent coefficients. It
reserves each user's final two interactions for the repository's outer
validation/test split, then uses the final interaction of the remaining
training history as an inner training target. A candidate user neighbor is
positive if its earlier history contains that target item. Reported pair
accuracy measures this proxy task, **not** MemRec's final recommendation
quality. Users without both positive and negative neighbors cannot train this
proxy and are counted in the coverage report.

## Evaluate in MemRec

Set the pruner section of an existing config to either checkpoint:

```yaml
memrec:
  pruner:
    mode: personalized_user
    checkpoint: results/user_neighbor_weights_books.json
```

Use `results/user_neighbor_weights_books.global.json` for the matched global
weight comparison, and `mode: llm_rules` for the original rule comparison.
Keep all other config settings and the evaluated user list the same. The
checkpoint files are under ignored `results/`; copy them elsewhere if they
must be included in version control.
