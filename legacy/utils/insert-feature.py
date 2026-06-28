# This script calculates the similarity ratio between two code snippets and inserts it as 'output' attribute in the dataset.

import json
import java_sim_exec_opt


def calculate_similarity(code1, code2):
    """
    Calculate the similarity ratio between two code snippets.
    """
    result = java_sim_exec_opt.similarity(code1, code2)
    return result

def process_dataset(input_file, output_file):
    """
    Process the dataset and insert similarity ratio as 'output' attribute.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    for entry in data:
        code1 = entry['code1']
        code2 = entry['code2']
        similarity_ratio = calculate_similarity(code1, code2)
        entry['output'] = similarity_ratio

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Input and Output file paths
input_file = 'data.json'
output_file = 'data2.json'

# Process dataset
process_dataset(input_file, output_file)
