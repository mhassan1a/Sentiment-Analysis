import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from models.LSTM import LSTMClassifier





def train_model(args):
    model = model.to(device)
    model.train()
    
    for epoch in range(EPOCHS):
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            output = model(input_ids, torch.tensor([len(input_ids[0])] * BATCH_SIZE).to(device))
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}')