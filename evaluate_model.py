import json
import requests
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

DATASET_PATH = '/Users/filipuhlarik/PycharmProjects/LabCafeFineTUNING/data/emotion_dataset_test.jsonl'

with open('./data/class_name_to_labels.json', 'r') as fh:
    class_name_to_labels = json.load(fh)
num_classes = len(class_name_to_labels)
# Load the dataset

with open(DATASET_PATH, 'r') as file:
    data = [json.loads(line) for line in file]


# Initialize the confusion matrix
confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

confusion_dict = {}
# Make POST requests to the API and update the confusion matrix
for record in tqdm.tqdm(data):
    response = requests.post('http://localhost:8534/classify', json={'text': record['text']})
    predicted_class = response.json()[0]['label']
    actual_class = record['label']

    if actual_class != predicted_class:
        name = actual_class + '_confused_as_' + predicted_class
        if name not in confusion_dict:
            confusion_dict[name] = []
        confusion_dict[name].append(record['text'])

    confusion_matrix[class_name_to_labels[actual_class]][class_name_to_labels[predicted_class]] += 1

# Display the confusion matrix
plt.figure(figsize=(10, 7))
sns.set(font_scale=1)  # for label size
sns.heatmap(confusion_matrix, annot=True, fmt='d', xticklabels=class_name_to_labels.keys(), yticklabels=class_name_to_labels.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the output as a vector image
plt.savefig("confusion_matrix.png", format='png')
plt.show()

# Create a list of tuples (actual, predicted, number_of_predictions)
predictions_list = []
for i in range(num_classes):
    for j in range(num_classes):
        if confusion_matrix[i][j] > 0:
            if i != j:
                predictions_list.append((list(class_name_to_labels.keys())[i], list(class_name_to_labels.keys())[j], confusion_matrix[i][j]))

# Sort the list by number_of_predictions
predictions_list.sort(key=lambda x: x[2], reverse=True)

# Output the sorted list
for prediction in predictions_list:
    print(prediction)

with open('confusion_dict.json', 'w') as fh:
    json.dump(confusion_dict, fh)

