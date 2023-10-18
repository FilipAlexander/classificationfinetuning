import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer

train_data_path = 'data/emotion_dataset_train.jsonl'
test_data_path = 'data/emotion_dataset_test.jsonl'
category_key = 'label'
text_containing_key = 'text'

tokenizer = AutoTokenizer.from_pretrained(
    'xlm-roberta-base', use_fast=True, add_prefix_space=True)

def raw_data_from_file(filepath):
    with open(filepath, 'r') as f:
        raw_data = [json.loads(line) for line in f]
    return raw_data

train_raw_data = raw_data_from_file(train_data_path)
test_raw_data = raw_data_from_file(test_data_path)

labelset = set()
for record in train_raw_data:
    for label in [record[category_key]]:
        labelset.add(label)

label_list = list(labelset)
class_name_to_labels = {label: i for i, label in enumerate(label_list)}

with open('./data/class_name_to_labels.json', 'w') as fh:
    json.dump(class_name_to_labels, fh)

num_labels = len(label_list)

def calculate_class_weights(transformed_dataset):
    # Extract the one-hot encoded labels from the dataset
    label_weights = {}
    label_count_dict = {}
    for row in transformed_dataset:
        if not row['labels'] in label_count_dict:
            label_count_dict[row['labels']] = 0
        label_count_dict[row['labels']] += 1

    total_labels = sum(label_count_dict.values())

    for label in label_count_dict:
        label_weights[label] = total_labels / label_count_dict[label]

    return label_weights

def transform_labels_to_tensors(raw_data):
    transformed_data = []
    for record in raw_data:
        for label in [record[category_key]]:
            record['labels'] = torch.tensor(class_name_to_labels[label])
            del record[category_key]
            transformed_data.append(record)
    return transformed_data

train_data = transform_labels_to_tensors(train_raw_data)
test_data = transform_labels_to_tensors(test_raw_data)

train_data = Dataset.from_list(train_data)
test_data = Dataset.from_list(test_data)


dataset_test = test_data.map(lambda example: tokenizer(example[text_containing_key], truncation=True, padding=False),
                                batched=True)
dataset_train = train_data.map(lambda example: tokenizer(example[text_containing_key], truncation=True, padding=False),batched=True)

dataset_test.save_to_disk(test_data_path.replace('.json', '_ready.json'))
dataset_train.save_to_disk(train_data_path.replace('.json', '_ready.json'))

class_weights = calculate_class_weights(dataset_train)

json.dumps(sorted(class_weights.items(),key=lambda x: x[0]),indent=4)
with open('./data/class_weights.json', 'w') as fh:
    json.dump(class_weights, fh, indent=4)
