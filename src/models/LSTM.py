import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, n_layers, dropout, embedding_weights):
        super(LSTMClassifier, self).__init__()
        
        self.vocab_size = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.embedding.weight = embedding_weights
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.num_classes)
        )
            
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(embedded, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    input_dim = 30522
    hidden_dim = 768
    num_classes = 5
    n_layers = 2
    dropout = 0.5
    embedding_weights = BertModel.from_pretrained("bert-base-uncased").embeddings.word_embeddings.weight
    model = LSTMClassifier(input_dim, hidden_dim, num_classes, n_layers, dropout, embedding_weights)
    x = torch.randint(0, input_dim, (32, 132))  
    print(f"Input size: {x.size()}") 
    y = torch.randint(0, num_classes, (32,))
    pred = model(x)
    pred_labels = torch.argmax(pred, dim=1)
    print(f"Predicted labels: {pred_labels}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
