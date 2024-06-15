from torch.utils.data import Dataset, DataLoader
import torch
from datasets import load_from_disk
from torch.utils.data import Subset
from numpy.random import permutation
from numpy import arange


class Yelp100Dataset(Dataset):
    def __init__(self):
        datasets1 = load_from_disk("data/yelp_tokenized.hf")
        idx = permutation(arange(100, 1000)).astype(int)
        self.dataset = datasets1['test'][idx[:100]]
        self.idx = idx[:100]
        self.encoding = self.dataset['input_ids']
        self.labels = self.dataset['label']
        self.mask = self.dataset['attention_mask']
        self.text = self.dataset['text']
        
    def __getitem__(self, idx):
        text = self.text[idx]
        token = torch.tensor(self.encoding[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        idx = self.idx[idx]
        return (idx, text, token, label)
    def __len__(self):
        return len(self.labels)
 
if __name__ == "__main__":
    dataset = Yelp100Dataset()
    print(f"length of train dataset: {len(dataset)}")
    for idx, text, input, label in dataset:
        print(idx, text, input.size(), label.size())
        break