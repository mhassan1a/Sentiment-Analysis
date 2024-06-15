
import numpy as np
import json
    
def load_from_disk(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def create_splits(data):
    train_data = np.array(data['train']['data'][:-2]).reshape(-1, 28)
    train_labels = np.array(data['train']['labels'][:-2]).reshape(-1)
    test_data = np.array(data['test']['data'][:-2]).reshape(-1, 28)
    test_labels = np.array(data['test']['labels'][:-2]).reshape(-1)
    return train_data, train_labels, test_data, test_labels