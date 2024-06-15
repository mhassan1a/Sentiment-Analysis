import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from models.LSTM import LSTMClassifier
from models.TRANSFORMER import TransformerClassifier
from local_datasets.yelp import YelpDataset
from local_datasets.goemotions import GoEmotionsDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import os
import argparse
from tqdm import tqdm
import json
from models.SVM import SVMClassifier
from models.RFC import RandomForestClassifier
import numpy as np

def load_model(model_path, device, model_name):
    if model_name == 'lstm':
        load = torch.load(model_path, map_location=torch.device(device))
        model = TransformerClassifier(**load['model_config'])
        model.load_state_dict(load['model'])
    elif model_name == 'transformer':
        load = torch.load(model_path, map_location=torch.device(device))
        model = TransformerClassifier(**load['model_config'])
        model.load_state_dict(load['model'])
    return model

def yelp_to_emotions(model_path, model_name, device):
    model = load_model(model_path, device, model_name)
    model.eval()
    model.to(device)
    splits = ['train', 'test']
    new_dataset = {}
    for split in splits:
        print(f"Processing {split} split")
        yelp = YelpDataset(split)
        yelp_loader = DataLoader(yelp, batch_size=512, shuffle=False)
        new_dataset[split] = {'data':[], 'labels':[]}
        for input, label, mask in tqdm(yelp_loader):
            input = input.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                output = model(input, mask)
            new_dataset[split]['data'].append(output.cpu().numpy().tolist())
            new_dataset[split]['labels'].append(label.cpu().numpy().tolist())
    return new_dataset
 
def train_svm(X_train, y_train, X_test, y_test):
    clf = SVMClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    return accuracy

def train_rf(X_train, y_train, X_test, y_test, n_jobs=8):
    clf = RandomForestClassifier(n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    return accuracy


def save_to_disk(data, path):
    with open(output_path, 'w') as f:
        json.dump(new_dataset, f)
    print("New dataset saved to disk")  
    
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

if __name__ == "__main__":
    path = 'final_models/best_eval_epoch__195876_model_dict_goemotions_trans_model.pt'
    model_name = 'transformer'
    output_path = 'data/yelp_to_emotions.json'  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    #new_dataset = yelp_to_emotions(path, model_name, device=device)
    #save_to_disk(new_dataset, output_path)
    new_dataset = load_from_disk(output_path)
    train_data, train_labels, test_data, test_labels = create_splits(new_dataset)
    accuracy = train_rf(train_data, train_labels, test_data, test_labels, n_jobs=32)
    print(f"Accuracy of RFC classifier: {accuracy}")