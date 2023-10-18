import random
from collections import defaultdict
import json
import sys
import os

def balance_dataset(filename):
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f]

    # Group by category
    category_to_items = defaultdict(list)
    for item in data:
        category_to_items[item['label']].append(item)

    # Find the maximum category count
    max_count = max(len(items) for items in category_to_items.values())

    # Oversample the undersampled classes
    balanced_data = []
    for category, items in category_to_items.items():
        oversampled_items = random.choices(items, k=max_count)
        balanced_data.extend(oversampled_items)

    # Save the balanced dataset alongside the original with a suffix _balanced before the extension
    base_filename, ext = os.path.splitext(filename)
    balanced_filename = f"{base_filename}_balanced{ext}"
    with open(balanced_filename, 'w') as f:
        for item in balanced_data:
            f.write(json.dumps(item) + '\n')

    return balanced_data

if __name__ == '__main__':
    balanced_data = balance_dataset(sys.argv[1])
    