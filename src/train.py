import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
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
from transformers import BertModel

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

def save_model(model, path, name):
    torch.save(model, path+name)    
    print(f"Model saved to: {path}")
    model_dict = dict(model=model.state_dict(),
                     model_name=model.__class__.__name__,
                     model_config=args.model_config)
    torch.save(model_dict, path + "model_dict_" + name )
    
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
    for input, labels, masks in data_loader:
        input = input.to(args.device)
        labels = labels.to(args.device)
        masks = masks.to(args.device)
        optimizer.zero_grad()
        output = model(input, masks)
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
    for input, labels, masks in dataloader:
        input = input.to(args.device)
        labels = labels.to(args.device)
        masks = masks.to(args.device)
        output = model(input, masks)
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
    for input, labels, masks in data_loader:
        input = input.to(args.device)
        labels = labels.to(args.device)
        masks = masks.to(args.device)
        output = model(input, masks)
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
            print("Saving model at epoch: ", epoch, "with validation accuracy: ", val_acc)
            save_model(model, os.path.join(args.output_path, f"best_eval_epoch__{args.job_id}_" ), args.output_name)
        
        if args.dry_run == 1:
            break        
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Learning rate too low. Stopping training")
            break
    test_loss, test_acc = test_model(args, model, loss_fn, test_loader)
    
    print("Test Loss: ", test_loss, "Test Accuracy: ", test_acc)
    save_model(model, os.path.join(args.output_path,  f"_{args.job_id}_"), args.output_name)
    save_results(args, training_losses, validation_losses, train_accs, val_accs, test_loss, test_acc, args.output_path)
    print("Model saved to: ", os.path.join(args.output_path), args.output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on Yelp or GoEmotions dataset')
    parser.add_argument('--dataset', type=str, default='goemotions', help='Dataset to train on (yelp or goemotions)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on (cpu or cuda)')
    parser.add_argument('--output_name', type=str, default='model.pt', help='Output model name')
    parser.add_argument('--output_path', type=str, default='checkpoints/', help='Output model path')
    parser.add_argument('--vocab_size', type=int, default=30522, help='Vocabulary size')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--lstm_hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--dry_run', type=int, default=1, help='Run a dry run')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--test', type=int, default=1, help='Run a test')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--model_name', type=str, default='transformer', help='Model to train (lstm or transformer)')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of heads')
    parser.add_argument('--job_id', type=int, default=0, help='Job id')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--trans_feedforward', type=int, default="2048", help='Feedforward dimension')
    parser.add_argument('--use_bert_embeddings', type=int, default=0, help='Learn embeddings')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer to use')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='Momentum')
    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.dataset.strip() == "yelp":
        weights = None
        args.output_name = "yelp_" + args.output_name
        num_classes = 5
        dataset = YelpDataset("train")
        idx = np.random.permutation(np.arange(len(dataset)))
        train_split = 0.8
        train_data = Subset(dataset, idx[:int(len(dataset) * train_split)])
        validate_data = Subset(dataset, idx[int(len(dataset) * train_split):])
        test_data = YelpDataset("test")
        
    elif args.dataset.strip() == "goemotions":
        weights = None
        args.output_name = "goemotions_" + args.output_name
        train_split = 0.7
        val_split = 0.15
        test_split = 0.15        
        num_classes = 28
        dataset = GoEmotionsDataset("train")
        idx = np.random.permutation(np.arange(len(dataset)))
        train_data = Subset(dataset, idx[:int(len(dataset) * train_split)])
        validate_data = Subset(dataset, idx[int(len(dataset) * train_split):int(len(dataset) * (train_split + val_split))])
        test_data = Subset(dataset, idx[int(len(dataset) * (train_split + val_split)):])
    else:
        raise ValueError("Dataset not supported")
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, 
                              shuffle=True, drop_last=True, num_workers=args.n_workers)
    validation_loader = DataLoader(validate_data, batch_size=args.batch_size,
                                   shuffle=False, drop_last=True, num_workers=args.n_workers)    
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.n_workers)
    if args.use_bert_embeddings == 1:
        args.embedding_dim = 768
        embedding_weights = BertModel.from_pretrained("bert-base-uncased").embeddings.word_embeddings.weight
    else:
        embedding_weights = None
    if args.model_name.strip() == "lstm":
        args.model_config = {
            "input_dim": args.vocab_size,
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.lstm_hidden_dim,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "num_classes": num_classes,
            "embedding_weights": embedding_weights,
        }
        model = LSTMClassifier(**args.model_config
                               ).to(args.device)
        
    elif args.model_name.strip() == "transformer":
        args.model_config = {
            "input_dim": args.vocab_size,
            "embedding_dim": args.embedding_dim,
            "num_classes": num_classes,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "trans_feedforward": args.trans_feedforward,
            "embedding_weights": embedding_weights,
        }
        model = TransformerClassifier(**args.model_config
                                      ).to(args.device)
    
    loss_fn = nn.CrossEntropyLoss(weight=weights).to(args.device)   
    if  args.optimizer.strip() == "adam":
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=args.lr, weight_decay=args.weight_decay) 
    elif args.optimizer.strip() == "sgd":
        optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                        momentum=args.momentum, lr=args.lr) 
    elif args.optimizer.strip() == "adamW":
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer not supported")    
    schedular = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__}  Number of parameters: {num_params}")
    train_model(args, model, loss_fn, optimizer, train_loader, validation_loader, test_loader, schedular)
