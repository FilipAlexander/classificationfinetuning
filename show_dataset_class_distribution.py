import json
import matplotlib.pyplot as plt
from collections import Counter
import sys

def load_and_visualize(filename):
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f]

    # Extract categories
    categories = [item['label'] for item in data]

    # Count the categories
    category_counts = Counter(categories)

    # Prepare data for the bar chart
    labels = list(category_counts.keys())
    values = list(category_counts.values())

    # Create bar chart
    print("Lenght of current dataset: ", len(data))
    plt.bar(labels, values)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title(f'{filename} Category Distribution')
    plt.xticks(rotation=90)
    plt.show()


if __name__ == '__main__':
    load_and_visualize(sys.argv[1])
