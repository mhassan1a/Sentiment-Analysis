import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

# Define constants
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-3

class YelpDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



def download_or_load_datasets():
    # Directory paths
    goemotion_path = "data/goemotion.hf"
    yelp_path = "data/yelp.hf"
    
    # Check if the data directory exists, create if it doesn't
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Load Yelp dataset
    if os.path.exists(yelp_path):
        print("Loading Yelp from disk")
        yelp = load_from_disk(yelp_path)
    else:
        print("Downloading Yelp dataset")
        yelp = load_dataset("Yelp/yelp_review_full")
        print("Saving Yelp to disk")
        yelp.save_to_disk(yelp_path)
    
    return yelp

def tokenize_datasets(yelp, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    # Tokenize Yelp dataset
    yelp = yelp.map(tokenize_function, batched=True)
    
    return yelp

def create_data_loader(yelp, batch_size):
    train_dataset = YelpDataset(yelp['train']['input_ids'], yelp['train']['label'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def train_model(model, data_loader, loss_fn, optimizer, device):
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

if __name__ == "__main__":
    yelp = download_or_load_datasets()
    
    # Load a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize the dataset
    yelp = tokenize_datasets(yelp, tokenizer)
    
    # Create data loader
    train_loader = create_data_loader(yelp, BATCH_SIZE)
    
    # Define model
    INPUT_DIM = tokenizer.vocab_size
    HIDDEN_DIM = 256
    OUTPUT_DIM = 5
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    
    model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    
    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, loss_fn, optimizer, device)

    