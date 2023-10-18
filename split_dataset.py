import json
import random
import os
import sys

def shuffle_and_split_dataset(filename, train_test_ratio, seed=42, max_samples=None):
    # Load the data
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f]
        if max_samples:
            data = data[:max_samples]

    # Shuffle the data
    random.seed(seed)
    random.shuffle(data)

    # Split the data
    split_index = int(len(data) * train_test_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    # Save the data
    base_filename, ext = os.path.splitext(filename)
    with open(f"{base_filename}_train{ext}", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    with open(f"{base_filename}_test{ext}", 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')

    return train_data, test_data

if __name__ == '__main__':
    shuffle_and_split_dataset(sys.argv[1], float(sys.argv[2]))
