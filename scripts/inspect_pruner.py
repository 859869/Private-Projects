"""Inspect the Books neighbor scores without calling an LLM.

Run ``python scripts/inspect_pruner.py`` for a tiny reproducible example, or
``python scripts/inspect_pruner.py --user-id 0`` for the local Books data.
"""
import argparse
import csv
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The package initializer also imports Torch-based modules. This inspection
# script needs only the graph, storage, and LLM rule pruner.
memory_package = types.ModuleType('src.memory')
memory_package.__path__ = [str(ROOT / 'src' / 'memory')]
sys.modules['src.memory'] = memory_package

from src.memory.graph import UserItemGraph
from src.memory.pruner_llm_rules import LLMRulePruner
from src.memory.storage import MemoryStorage


def example_dataset():
    metadata = {
        1: {'title': 'Galactic Atlas', 'description': 'A guide to stars and galaxies'},
        2: {'title': 'Galactic Atlas Volume Two', 'description': 'A guide to stars and galaxies'},
        3: {'title': 'Garden Cookbook', 'description': 'Recipes for fresh vegetables'},
        4: {'title': 'Galactic Atlas Companion', 'description': 'A guide to stars and galaxies'},
    }
    return SimpleNamespace(item_metadata=metadata, train_data={10: [1, 2, 3], 20: [2, 4]})


def books_dataset(data_dir: Path, user_id: int):
    events = defaultdict(list)
    with (data_dir / 'instructrec-books.inter').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            events[int(row['user_id'])].append((int(row['timestamp']), int(row['item_id'])))
    train_data = {
        uid: [item_id for _, item_id in sorted(items)[:-2]]
        for uid, items in events.items() if len(items) >= 3
    }
    if user_id not in train_data:
        raise ValueError(f'User {user_id} has no training history in {data_dir}')

    needed_ids = set(train_data[user_id])
    metadata = {}
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with (data_dir / 'instructrec-books.meta').open(encoding='utf-8', errors='replace', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            item_id = int(row['item_id'])
            if item_id in needed_ids:
                metadata[item_id] = {
                    'title': row.get('title', ''),
                    'description': (row.get('description') or '')[:300],
                }
                if len(metadata) == len(needed_ids):
                    break
    return SimpleNamespace(item_metadata=metadata, train_data=train_data)


def load_saved_memories(path: Path, storage: MemoryStorage, user_id: int, graph: UserItemGraph):
    neighbor_data = graph.get_user_neighbors(user_id, max_items=100, max_users=50)
    wanted_users = {user_id} | {uid for uid, _ in neighbor_data['user_neighbors']}
    wanted_items = {iid for iid, _ in neighbor_data['item_neighbors']}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            entry = json.loads(line)
            if entry.get('type') == 'user' and entry.get('id') in wanted_users:
                storage.update_user_memory(entry['id'], entry.get('memory', ''))
            elif entry.get('type') == 'item' and entry.get('id') in wanted_items:
                storage.update_item_memory(entry['id'], entry.get('memory', ''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user-id', type=int, help='Inspect a user in the local Books dataset')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'processed' / 'instructrec-books')
    parser.add_argument('--user-memory', help='Optional current user memory text')
    parser.add_argument('--memory-file', type=Path, help='Optional saved memory.jsonl from a prior run')
    parser.add_argument('--k', type=int, default=16)
    args = parser.parse_args()

    dataset = example_dataset() if args.user_id is None else books_dataset(args.data_dir, args.user_id)
    user_id = 10 if args.user_id is None else args.user_id
    graph = UserItemGraph(dataset)
    storage = MemoryStorage()
    storage.initialize_item_descriptions(dataset.item_metadata)
    if args.user_id is None:
        storage.update_user_memory(10, 'Enjoys stars and galaxies')
        storage.update_user_memory(20, 'Reads about stars and galaxies')
    if args.memory_file:
        load_saved_memories(args.memory_file, storage, user_id, graph)
    if args.user_memory:
        storage.update_user_memory(user_id, args.user_memory)

    pruner = LLMRulePruner(
        'instructrec-books', k=args.k, dataset=dataset, memory_storage=storage,
    )
    result = pruner.prune(user_id, graph)
    print(f"\nUser {user_id}: {result['n_items']} item neighbors, {result['n_users']} user neighbors")
    print('type  id       base   content  memory  final   title')
    for neighbor in result['neighbors']:
        item_id = neighbor['id']
        title = dataset.item_metadata.get(item_id, {}).get('title', '') if neighbor['type'] == 'item' else ''
        final_score = pruner.score_neighbor(neighbor)
        print(
            f"{neighbor['type']:<5} {item_id:<8} {neighbor['score']:>5.3f} "
            f"{neighbor['metadata_overlap']:>7.3f} {neighbor['memory_sim']:>7.3f} "
            f"{final_score:>6.3f}  {title[:55]}"
        )
    if storage.get_user_memory(user_id) is None:
        print('\nNo user memory is available, so memory similarity is 0 for this inspection.')


if __name__ == '__main__':
    main()
