import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
#
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, 
                 num_classes, n_heads, n_layers, dropout, dim_feedforward, embedding_weihgts=None):
        super(TransformerClassifier, self).__init__()
    
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        if embedding_weihgts is not None:
            self.embedding.weight = nn.Parameter(embedding_weihgts, requires_grad=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, bias=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=n_layers,
                                                         norm=nn.LayerNorm(embedding_dim))

        self.fc = nn.Linear(embedding_dim, num_classes, bias=False)

    def forward(self, x, mask):
        embedded = self.embedding(x)  
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
    pred_labels = torch.argmax(pred, dim=1)
    
    print(f"Predicted labels: {pred_labels}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
