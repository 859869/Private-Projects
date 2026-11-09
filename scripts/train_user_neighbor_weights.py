"""Train a user-neighbor weight gate without using validation or test interactions.

Example: python scripts/train_user_neighbor_weights.py --max-users 5000
"""

import argparse
import csv
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The training gate needs NumPy only. The memory package initializer imports
# the Torch-backed baseline pruner, which is unrelated to this offline job.
memory_package = types.ModuleType("src.memory")
memory_package.__path__ = [str(ROOT / "src" / "memory")]
sys.modules["src.memory"] = memory_package

from src.memory.graph import UserItemGraph
from src.memory.personalized_weights import UserWeightModel
from src.memory.personalized_weights.training import build_pairs, fit_gate, pair_metrics


def load_data(data_dir, dataset_name):
    histories = {}
    with (data_dir / f"{dataset_name}.inter").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            histories.setdefault(int(row["user_id"]), []).append(
                (float(row["timestamp"]), int(row["item_id"])))
    # Match RecDataset's outer leave-one-out split. The inner last training
    # interaction is a label and is omitted from every graph history.
    original_train = {user_id: [item for _, item in sorted(events, key=lambda event: event[0])[:-2]]
                      for user_id, events in histories.items() if len(events) >= 3}
    targets = {user_id: items[-1] for user_id, items in original_train.items() if len(items) >= 2}
    prefixes = {user_id: (items[:-1] if user_id in targets else items)
                for user_id, items in original_train.items()}
    needed_items = {item for items in prefixes.values() for item in items}
    metadata = {}
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with (data_dir / f"{dataset_name}.meta").open(encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            item_id = int(row["item_id"])
            if item_id in needed_items:
                metadata[item_id] = {"title": row.get("title", "")}
    return SimpleNamespace(train_data=prefixes, item_metadata=metadata), targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="instructrec-books")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "user_neighbor_weights.json")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data_dir = args.data_dir or ROOT / "data" / "processed" / args.dataset
    dataset, targets = load_data(data_dir, args.dataset)
    graph = UserItemGraph(dataset)
    records, coverage = build_pairs(dataset, graph, targets, max_users=args.max_users, seed=args.seed)
    if not records:
        raise RuntimeError(f"No training pairs: {coverage}")
    rng = np.random.default_rng(args.seed)
    users = np.asarray(sorted({row[0] for row in records}))
    rng.shuffle(users)
    n_valid = max(1, int(0.2 * len(users)))
    valid_users = set(users[:n_valid])
    train_records = [row for row in records if row[0] not in valid_users]
    valid_records = [row for row in records if row[0] in valid_users]
    if not train_records:
        raise RuntimeError("Need at least two covered users for train/validation split")
    global_model = fit_gate(train_records, valid_records, personalized=False, epochs=args.epochs)
    user_model = fit_gate(train_records, valid_records, personalized=True, epochs=args.epochs,
                          initial_model=global_model)
    report = {"coverage": coverage, "train_users": len(users) - n_valid,
              "valid_users": n_valid, "global": pair_metrics(global_model, valid_records),
              "personalized": pair_metrics(user_model, valid_records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    user_model.save(args.output, metadata={"dataset": args.dataset, "training_report": report})
    global_output = args.output.with_name(args.output.stem + ".global" + args.output.suffix)
    global_model.save(global_output, metadata={"dataset": args.dataset, "training_report": report})
    print(json.dumps({"checkpoint": str(args.output), "global_checkpoint": str(global_output),
                      **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
