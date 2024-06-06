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

def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_cpu_numpy(x):
    return x.detach().cpu().numpy()

def save_model(model, path):
    torch.save(model.state_dict(), path)    
    
def save_results(training_losses, validation_losses, train_acc, val_acc, path):
    with open(os.path.join(path, "training_losses.json"), "w") as f:
        json.dump(training_losses, f)
        print("Training losses saved to: ", os.path.join(path, "training_losses.json"))
    with open(os.path.join(path, "validation_losses.json"), "w") as f:
        json.dump(validation_losses, f)
        print("Validation losses saved to: ", os.path.join(path, "validation_losses.json"))
    with open(os.path.join(path, "train_acc.json"), "w") as f:
        json.dump(train_acc, f)
        print("Training accuracy saved to: ", os.path.join(path, "train_acc.json"))
    with open(os.path.join(path, "val_acc.json"), "w") as f:
        json.dump(val_acc, f)
        print("Validation accuracy saved to: ", os.path.join(path, "val_acc.json"))
    
def training_step(args, model, loss_fn, optimizer, data_loader):
    model.train()
    for input, labels in data_loader:
        print(input.size(), labels.size())
        input = input.to(args.device)
        labels = labels.to(args.device)
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == labels).float().mean()
        if args.dry_run:
            break
    return model, loss, acc

@torch.no_grad()
def validate_step(args, model, dataloader, loss_fn):
    correct = 0
    total = 0
    with torch.no_grad():
        for input, labels in dataloader:
            input_ids = input.to(args.device)
            labels = labels.to(args.device)
            output = model(input_ids)
            loss = loss_fn(output, labels)
            acc = (output.argmax(dim=1) == labels).float().sum()
            if args.dry_run:
                break
    return loss, acc

def train_model(args, model, loss_fn, optimizer, train_loader, validation_loader, schedular):
    print("START: Training model:.....")
    training_losses = []
    validation_losses = []
    val_accs = []
    train_accs = []
    best_val_acc = 0
    progressbar = tqdm(range(args.epochs), desc='Epochs')
    for epoch in progressbar:
        model, loss, acc = training_step(args, model, loss_fn, optimizer, train_loader)
        training_losses.append(to_cpu_numpy(loss).tolist())
        train_accs.append(to_cpu_numpy(acc).tolist())
        
        val_loss, val_acc = validate_step(args, model, validation_loader, loss_fn)
        schedular.step(val_loss)
        validation_losses.append(to_cpu_numpy(val_loss).tolist())
        val_accs.append(to_cpu_numpy(val_acc).tolist())
        
        
        progressbar.set_postfix({'Training Loss': training_losses[-1],
                                 'Validation Loss': validation_losses[-1]})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, os.path.join(args.output_path, f"best_eval_epoch_{epoch}_"+args.output_name))
            
        if args.dry_run:
            break        
    save_model(model, os.path.join(args.output_path, args.output_name))
    save_results(training_losses, validation_losses, train_accs, val_accs, args.output_path)
    
    print("END: Training model:.....")
    print("Model saved to: ", os.path.join(args.output_path, args.output_name))



datasets_dic = {"yelp": YelpDataset, "goemotions": GoEmotionsDataset}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on Yelp or GoEmotions dataset')
    parser.add_argument('--dataset', type=str, default='goemotions', help='Dataset to train on (yelp or goemotions)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on (cpu or cuda)')
    parser.add_argument('--output_name', type=str, default='model.pt', help='Output model name')
    parser.add_argument('--output_path', type=str, default='checkpoints/', help='Output model path')
    parser.add_argument('--vocab_size', type=int, default=30522, help='Vocabulary size')
    parser.add_argument('--emb_dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dry_run', type=bool, default=True, help='Run a dry run')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset.strip() == "yelp":
        args.output_name = "yelp_" + args.output_name
        num_classes = 5
        train_split = 0.8
        dataset = YelpDataset("train")
        train_data = dataset[:int(len(dataset) * train_split)]
        validate_data = dataset[int(len(dataset) * train_split):]
        
    elif args.dataset.strip() == "goemotions":
        args.output_name = "goemotions_" + args.output_name
        num_classes = 28
        train_data = GoEmotionsDataset("train")
        validate_data = GoEmotionsDataset("validation")
    else:
        raise ValueError("Dataset not supported")
    
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validate_data, batch_size=args.batch_size, shuffle=False, drop_last=True)    
    loss_fn = nn.CrossEntropyLoss()
    
    vocab_size = args.vocab_size
    emb_dim = args.emb_dim
    dropout = args.dropout
    n_layers = args.n_layers
    lr = args.lr
    
    model = LSTMClassifier(vocab_size, emb_dim, num_classes, n_layers, dropout)
    optimizer = Adam(model.parameters(), lr)
    schedular = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    seed_everything(args.seed)
    train_model(args, model, loss_fn, optimizer, train_loader, validation_loader, schedular)
    