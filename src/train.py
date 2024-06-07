import torch
import torch.nn as nn
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
import numpy as np
from models.TRANSFORMER import TransformerClassifier
from torch.utils.data import Subset

def seed_everything(seed):
    np.random.seed(seed)
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

def save_results(args, training_losses, validation_losses, train_acc, val_acc, test_loss, test_acc, path):
    results = {
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "args": vars(args),
    }
    with open(os.path.join(path, f"results_{args.job_id}_.json"), "w") as f:
        json.dump(results, f)
    print(f"Results saved to: {os.path.join(path, f'results_{args.job_id}_.json')}")

def training_step(args, model, loss_fn, optimizer, data_loader):
    model.train()
    training_loss = 0
    correct_predictions = 0
    total_predictions = 0
    for input, labels in data_loader:
        input = input.to(args.device)
        labels = labels.to(args.device)
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        correct_predictions += (output.argmax(dim=1) == labels).sum().item()
        total_predictions += labels.size(0)
        if args.dry_run == 1:
            break
    accuracy = correct_predictions / total_predictions
    return model, training_loss / len(data_loader), accuracy

@torch.no_grad()
def validate_step(args, model, dataloader, loss_fn):
    model.eval()
    validation_loss = 0
    correct_predictions = 0
    total_predictions = 0
    for input, labels in dataloader:
        input = input.to(args.device)
        labels = labels.to(args.device)
        output = model(input)
        loss = loss_fn(output, labels)
        validation_loss += loss.item()
        correct_predictions += (output.argmax(dim=1) == labels).sum().item()
        total_predictions += labels.size(0)
        if args.dry_run == 1:
            break
    accuracy = correct_predictions / total_predictions
    return validation_loss / len(dataloader), accuracy

@torch.no_grad()
def test_model(args, model, loss_fn, data_loader):
    model.eval()
    test_loss = 0
    correct_predictions = 0
    total_predictions = 0
    for input, labels in data_loader:
        input = input.to(args.device)
        labels = labels.to(args.device)
        output = model(input)
        loss = loss_fn(output, labels)
        test_loss += loss.item()
        correct_predictions += (output.argmax(dim=1) == labels).sum().item()
        total_predictions += labels.size(0)
        if args.dry_run == 1:
            break
    accuracy = correct_predictions / total_predictions
    return test_loss / len(data_loader), accuracy

def train_model(args, model, loss_fn, optimizer, train_loader, validation_loader, test_loader, schedular):
    training_losses = []
    validation_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    progressbar = tqdm(range(args.epochs), desc='Epochs')
    for epoch in progressbar:
        model, train_loss, train_acc = training_step(args, model, loss_fn, optimizer, train_loader)
        training_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_loss, val_acc = validate_step(args, model, validation_loader, loss_fn)
        schedular.step(val_loss)
        validation_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch: {epoch} Training Loss: {train_loss} Validation Loss: {val_loss} lr: {optimizer.param_groups[0]['lr']}")
        print(f"Epoch: {epoch} Training Accuracy: {train_acc} Validation Accuracy: {val_acc}")
        
        progressbar.set_postfix({'Training Loss': train_loss, 'Validation Loss': val_loss, "lr": 
                                optimizer.param_groups[0]['lr']})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, os.path.join(args.output_path, f"best_eval_epoch__{args.job_id}_" + args.output_name))
        
        if args.dry_run == 1:
            break        
    test_loss, test_acc = test_model(args, model, loss_fn, test_loader)
    
    print("Test Loss: ", test_loss, "Test Accuracy: ", test_acc)
    save_model(model, os.path.join(args.output_path, args.output_name))
    save_results(args, training_losses, validation_losses, train_accs, val_accs, test_loss, test_acc, args.output_path)
    print("Model saved to: ", os.path.join(args.output_path, args.output_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on Yelp or GoEmotions dataset')
    parser.add_argument('--dataset', type=str, default='goemotions', help='Dataset to train on (yelp or goemotions)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on (cpu or cuda)')
    parser.add_argument('--output_name', type=str, default='model.pt', help='Output model name')
    parser.add_argument('--output_path', type=str, default='checkpoints/', help='Output model path')
    parser.add_argument('--vocab_size', type=int, default=30522, help='Vocabulary size')
    parser.add_argument('--emb_dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--dry_run', type=int, default=1, help='Run a dry run')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--test', type=int, default=1, help='Run a test')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--model_name', type=str, default='lstm', help='Model to train (lstm or transformer)')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of heads')
    parser.add_argument('--job_id', type=int, default=0, help='Job id')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    args = parser.parse_args()
    print(args)
    
    seed_everything(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.dataset.strip() == "yelp":
        args.output_name = "yelp_" + args.output_name
        num_classes = 5
        dataset = YelpDataset("train")
        idx = np.random.permutation(np.arange(len(dataset)))
        train_split = 0.8
        train_data = Subset(dataset, idx[:int(len(dataset) * train_split)])
        validate_data = Subset(dataset, idx[int(len(dataset) * train_split):])
        test_data = YelpDataset("test")
        
    elif args.dataset.strip() == "goemotions":
        args.output_name = "goemotions_" + args.output_name
        num_classes = 28
        train_data = GoEmotionsDataset("train")
        validate_data = GoEmotionsDataset("validation")
        test_data = GoEmotionsDataset("test")
    else:
        raise ValueError("Dataset not supported")
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, 
                              shuffle=True, drop_last=True, num_workers=args.n_workers)
    validation_loader = DataLoader(validate_data, batch_size=args.batch_size,
                                   shuffle=False, drop_last=True, num_workers=args.n_workers)    
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.n_workers)
    
    
    if args.model_name.strip() == "lstm":
        model = LSTMClassifier(args.vocab_size, args.emb_dim,
                               num_classes, args.n_layers, args.dropout).to(args.device)
    elif args.model_name.strip() == "transformer":
        model = TransformerClassifier(args.vocab_size, args.emb_dim,
                                      num_classes, args.n_heads, args.n_layers, args.dropout).to(args.device)
    
    loss_fn = nn.CrossEntropyLoss().to(args.device)    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedular = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    train_model(args, model, loss_fn, optimizer, train_loader, validation_loader, test_loader, schedular)
