from torch.utils.data import Dataset, DataLoader
import torch
from datasets import load_from_disk
from torch.utils.data import Subset
from numpy.random import permutation
from numpy import arange

class YelpDataset(Dataset):
    def __init__(self,  split):
        datasets = load_from_disk("data/yelp_tokenized.hf")
        self.dataset = datasets[split]
        self.encoding = self.dataset['input_ids']
        self.labels = self.dataset['label']
        self.mask = self.dataset['attention_mask']
        
    def __getitem__(self, idx):
        token = torch.tensor(self.encoding[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        mask = torch.tensor(self.mask[idx], dtype=torch.bool)
        return (token, label, mask)

    def __len__(self):
        return len(self.labels)
    
    
if __name__ == "__main__":
    yelp_train = YelpDataset("train")
    yelp_train_loader = DataLoader(yelp_train, batch_size=32, shuffle=False)
    dataset = YelpDataset("train")
    idx = permutation(arange(len(dataset)))
    train_split = 0.8
    train_data = Subset(dataset, idx[:int(len(dataset) * train_split)])
    validate_data = Subset(dataset, idx[int(len(dataset) * train_split):])
    
    print(f"length of train dataset: {len(train_data)}")
    print(f"length of validation dataset: {len(validate_data)}")
    for input, label, mask in yelp_train_loader:
        print(input.size(), label.size(), mask.size())
        break
    