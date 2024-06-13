import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import math
from torch import Tensor


class PositionalEncoding(nn.Module):
    """ Positional encoding for transformer model
    https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch

    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, 
                 num_classes, n_heads, n_layers, dropout, 
                 trans_feedforward, embedding_weights=None):
        super(TransformerClassifier, self).__init__()
        
        
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        if embedding_weights is not None:
            self.embedding.weight = nn.Parameter(embedding_weights, requires_grad=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads,
            dim_feedforward=trans_feedforward, dropout=dropout, bias=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=n_layers,
                                                         norm=nn.LayerNorm(embedding_dim))

        self.fc = nn.Linear(embedding_dim, num_classes, bias=False)
        self.positional_encoding = PositionalEncoding(embedding_dim)

    def forward(self, x, mask):
        embedded = self.embedding(x) + self.positional_encoding(self.embedding(x))
        embedded = embedded.permute(1, 0, 2)  
        transformer_out = self.transformer_encoder(embedded)
        out = transformer_out.mean(dim=0)  
        out = self.fc(out)
        return out

if __name__ == "__main__":
    input_dim = 30522 
    embedding_dim = 768  
    num_classes = 28
    n_heads = 8
    n_layers = 3
    dropout = 0.1
    embedding_weights = BertModel.from_pretrained("bert-base-uncased").embeddings.word_embeddings.weight
    model = TransformerClassifier(input_dim, embedding_dim,
                                  num_classes, n_heads, n_layers, 
                                  dropout, 1024,
                                  embedding_weights)
    
    x = torch.randint(0, input_dim, (32, 132))  
    print(f"Input size: {x.size()}") 
    y = torch.randint(0, num_classes, (32,))
    MASK = torch.zeros(size=(32,132), dtype=torch.bool)
    print(f"Mask size: {MASK.size()}")
    print(f"Labels size: {y.size()}")
    pred = model(x, MASK)
    print(f"Output size: {pred.size()}")
    pred_labels = torch.argmax(pred, dim=1)
    
    print(f"Predicted labels: {pred_labels}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
