import json

def transform_dataset(dataset_path):
    label_mapping = {
        '0': 'sadness',
        '1': 'joy',
        '2': 'love',
        '3': 'anger',
        '4': 'fear',
        '5': 'surprise'
    }
    transformed_dataset = []
    with open(dataset_path, 'r') as fh:
        dataset = fh.readlines()
        for example in dataset:
            example = json.loads(example)
            text = example['text']
            label = label_mapping[str(example['label'])]
            transformed_example = {'text': text, 'label': label}
            transformed_dataset.append(transformed_example)
    return transformed_dataset

dataset_train = transform_dataset('./data/data_emotion.jsonl')
with open('./data/emotion_dataset.jsonl', 'w') as fh:
    for example in dataset_train:
        json.dump(example, fh)
        fh.write('\n')
