import json
from sklearn.model_selection import train_test_split

# Load your dataset from a file
with open('data.json', 'r') as file:
    dataset = json.load(file)

# Shuffle the dataset to ensure it's randomly distributed
import random
random.shuffle(dataset)

# Split the dataset
# We're using a split of 70% for training, 15% for validation, and 15% for testing
train, temp = train_test_split(dataset, test_size=0.3, random_state=42)
validation, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save the splits into separate files
with open('training.json', 'w') as f:
    json.dump(train, f, indent=4)

with open('validation.json', 'w') as f:
    json.dump(validation, f, indent=4)

with open('test.json', 'w') as f:
    json.dump(test, f, indent=4)
