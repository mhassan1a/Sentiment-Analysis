import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from models.LSTM import LSTMClassifier
from local_datasets.yelp import YelpDataset
from local_datasets.goemotions import GoEmotionsDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import os
import argparse
from tqdm import tqdm
import json


def load_model(model_path, model):
    model.load_state_dict(torch.load(model_path))
    return model

def predict(model, data_loader, device):
    model.eval()
    predictions = []
    for input, labels in data_loader:
        input = input.to(device)
        output = model(input)
        predictions.append(output.argmax(dim=1))
    return predictions

def raw_predict(model, data_loader, device):
    model.eval()
    predictions = []
    for input, labels in data_loader:
        input = input.to(device)
        output = model(input)
        predictions.append(output)
    return predictions, labels

def evaluate(predictions, labels):
    correct = 0
    for pred, label in zip(predictions, labels):
        correct += (pred.argmax(dim=1) == label).float().sum()
    accuracy = correct / len(labels)
    return accuracy

def main(args):
    if args.dataset == "yelp":
        dataset = YelpDataset(args.split)
    elif args.dataset == "goemotions":
        dataset = GoEmotionsDataset(args.split)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = LSTMClassifier(30522, 256, 5, 10, 0.5)
    model = load_model(args.model_path, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predictions = predict(model, data_loader, device)
    accuracy = evaluate(predictions, dataset.labels)
    print(f"Accuracy: {accuracy}")